# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 17:44
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : demo.py
# @Software: PyCharm
import os
import sys
import argparse
import cv2
import mmcv
import torch
import numpy as np
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
# recog
import torch.nn.functional as F
from gesture_recog.build_model import build_model, get_test_transformer
from metric.postprocess import postprocessing


def extract_gesture(src_path, save_path, types, gpu_id, is_contain_imgwh=True, **recog_kwarg):
    config = recog_kwarg["config"]
    model = recog_kwarg["model"]
    weight_path = recog_kwarg["weight_path"]
    gesture_model = build_model(config, model, weight_path)
    gesture_model.to("cuda:{}".format(gpu_id))
    trans = get_test_transformer(config)
    with open(src_path, "r", encoding="utf-8") as fr:
        with open(save_path, "w", encoding="utf-8") as fw:
            for idx, line in enumerate(fr.readlines()):
                if idx % 1000 == 0:
                    print("recog deal {}".format(idx))
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                if img_path == "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/crop/imgs/ges30_imgs_1018_1019/1/_VID_20200412_130358/188.jpg":
                    print("img:{}".format(img_path))
                label = int(line_sp[1])
                if is_contain_imgwh:
                    box = list(map(float, line_sp[4:8]))  # x, y, w, h
                    img_w, img_h = int(line_sp[2]), int(line_sp[3])
                else:
                    box = list(map(float, line_sp[2:6]))  # x, y, w, h

                kpts = None
                img = cv2.imread(img_path)
                img_new, bbox_new, kpts, img_path = trans(img, box, kpts, img_path)
                img_new = img_new.unsqueeze(0)
                img_new = img_new.to(gpu_id)
                ret = gesture_model(img_new, mode="test")
                if isinstance(ret, tuple):
                    if types == "rodd":
                        feature, logits = ret[0], ret[1]
                        conf = postprocessing(logits, types, **recog_kwarg)
                        feature = F.softmax(feature, dim=1)
                        feature = feature.cpu().detach().numpy()
                        pred = feature.argmax(axis=1)[0]
                        sco = feature.max(axis=1)[0]
                    elif types == "arcface":
                        feature, logits = ret
                        conf = postprocessing(logits, "rodd", **recog_kwarg)
                        first_sing_vec_path = recog_kwarg["first_sing_vec_path"]
                        first_sing_vecs = np.load(first_sing_vec_path)  # shape [num_cls, fea_dim]
                        first_sing_vecs = torch.from_numpy(first_sing_vecs).to(feature.device)
                        cos_sim = torch.matmul(F.normalize(feature, dim=1, p=2),
                                               F.normalize(first_sing_vecs, dim=1, p=2).T)
                        cos_sim = cos_sim.cpu().detach().numpy()
                        pred = cos_sim.argmax(axis=1)[0]
                        sco = cos_sim.max(axis=1)[0]
                    else:
                        ret = ret[0]
                        conf = postprocessing(ret, types, **recog_kwarg)
                        if isinstance(ret, torch.Tensor):
                            ret = F.softmax(ret, dim=1)
                            ret = ret.cpu().detach().numpy()
                        pred = ret.argmax(axis=1)[0]
                        sco = ret.max(axis=1)[0]
                else:
                    conf = postprocessing(ret, types, **recog_kwarg)
                    if isinstance(ret, torch.Tensor):
                        ret = F.softmax(ret, dim=1)
                        ret = ret.cpu().detach().numpy()
                    pred = ret.argmax(axis=1)[0]
                    sco = ret.max(axis=1)[0]
                box_str = " ".join(list(map(str, box)))
                if is_contain_imgwh:
                    new_line = img_path + " " + str(pred) + " " + str(img_w) + " " + str(
                        img_h) + " " + box_str + " " + "{:.2f}".format(sco) + " " + "{:.2f}".format(conf[0]) + "\n"
                else:
                    new_line = img_path + " " + str(pred) + " " + box_str + " " + "{:.2f}".format(
                        sco) + " " + "{:.2f}".format(conf[0]) + "\n"
                # print(new_line)
                fw.write(new_line)
                fw.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--recog_cfg', default="tools/gesture_recog/exp_rodd_cls7/config.py",
                        help='test config file path')
    parser.add_argument('--recog_model', default="gesture_recog.exp_rodd_cls7.model.Model",
                        help='test config file path')
    parser.add_argument('--recog_ckp',
                        default="/zhoudu/checkpoints/gesture_recog/exp_rodd_cls7/checkpoint/model-epoch189.weights",
                        help='test config file path')
    parser.add_argument('--recog_types', default="rodd", help='msp proser rodd')
    parser.add_argument('--gpu_id', default=1, type=int, help='gpu id')
    parser.add_argument('--first_sing_vec_path',
                        default="/zhoudu/test/gesture/feas/exp_rodd_cls7/first_sing_vec.train.ges30_cloudwalk.ges30_imgs_1015_1017.npy",
                        help='when recog_types is rodd, it will use')
    parser.add_argument('--src_path',
                        default="/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/det.test.ges30_imgs_1018_1019.map.box.txt",
                        help='src list path')
    parser.add_argument('--save_path',
                        default="/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/det.recog.test.ges30_imgs_1018_1019.map.box.txt",
                        help='save result path')
    args = parser.parse_args()

    # 构建识别模型
    config = args.recog_cfg
    model = args.recog_model
    weight_path = args.recog_ckp
    types = args.recog_types
    gpu_id = args.gpu_id
    first_sing_vec_path = args.first_sing_vec_path
    src_path = args.src_path
    save_path = args.save_path

    recog_kwarg = {
        "config": config,
        "model": model,
        "weight_path": weight_path,
        "first_sing_vec_path": first_sing_vec_path
    }

    extract_gesture(src_path, save_path, types, gpu_id, **recog_kwarg)
