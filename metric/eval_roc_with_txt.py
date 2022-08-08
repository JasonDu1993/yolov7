# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 14:04
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : eval.py
# @Software: PyCharm
import os
import sys
import mmcv
import numpy as np
import cv2
import argparse
import shutil
from collections import defaultdict
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.box_and_kpt_utils import IOU_XYWH
from utils.draw_box_kpt_utils import draw_box_and_kpt, show_img, save_img
import matplotlib.pyplot as plt
from utils.datas.gesture.class_color_c9 import CLASSES, PALETTE
from metric.roc import get_roc_with_err_rates, get_fprs_with_tprs
from utils.plot_curve import plot_curve
from utils.datas.gesture.merge import get_merge_txt_seq
from metric.postprocess import postprocessing
from utils.box_and_kpt_utils import xyxy2xywh, get_area


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def eval(label_path, vis=0, offset=0, sco_thd=0.5, img_root="/zhoudu/test/gesture/show/yolov3_416_23/imgs",
         txt_root="/zhoudu/test/gesture/show/yolov3_416_23/cluster_rst/", txt_prefix="", use_recog_model=False,
         use_gt_box=False, roc_path=None, types="msp", **kwargs):
    gts = defaultdict(list)
    gt_box_num = 0
    with open(label_path, "r", encoding="utf-8") as fr:
        for idx, line in enumerate(fr.readlines()):
            gt_box_num += 1
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            label = int(line_sp[1])
            box = list(map(float, line_sp[4:8]))  # x, y, w, h
            img_w, img_h = int(line_sp[2]), int(line_sp[3])
            data = {"img_path": img_path, "label": label, "box": box, "img_w": img_w, "img_h": img_h}
            gts[img_path].append(data)

    if use_recog_model:
        config = kwargs["config"]
        model = kwargs["model"]
        weight_path = kwargs["weight_path"]
        gesture_model = build_model(config, model, weight_path)
        gpu = 1
        gesture_model.to("cuda:{}".format(gpu))
        trans = get_test_transformer(config)
    y_true = []
    y_score = []
    acc_pos = 0
    pos_num = 0
    acc_neg = 0
    neg_num = 0
    for idx, img_path in enumerate(gts):
        gt = gts[img_path]
        max_area = -1
        reserve_data = None
        cnt = 0
        for gt_data in gt:
            gt_box = gt_data["box"]
            area = get_area(gt_box)
            if area > max_area:
                cnt += 1
                # if cnt >= 2:
                # print("idx:{} box_num:{} img_path:{}".format(idx, cnt, img_path))
                max_area = area
                reserve_data = gt_data
        img_path = reserve_data["img_path"]
        gt_label = reserve_data["label"]
        gt_box = reserve_data["box"]
        img = cv2.imread(img_path)

        if use_recog_model:
            kpts = None
            img_new, bbox_new, kpts, img_path = trans(img, gt_box, kpts, img_path)
            img_new = img_new.unsqueeze(0)
            img_new = img_new.to(gpu)
            ret = gesture_model(img_new, mode="test")
            if isinstance(ret, tuple):
                if types == "rodd":
                    feature, logits = ret[0], ret[1]
                    conf = postprocessing(logits, types, **recog_kwarg)
                    y_score.append(conf[0])
                    feature = F.softmax(feature, dim=1)
                    feature = feature.cpu().detach().numpy()
                    pred = feature.argmax(axis=1)[0]
                    sco = feature.max(axis=1)[0]
                elif types == "arcface":
                    feature, logits = ret
                    conf = postprocessing(logits, "rodd", **recog_kwarg)
                    y_score.append(conf[0])
                    first_sing_vec_path = recog_kwarg["first_sing_vec_path"]
                    first_sing_vecs = np.load(first_sing_vec_path)  # shape [num_cls, fea_dim]
                    first_sing_vecs = torch.from_numpy(first_sing_vecs).to(feature.device)
                    cos_sim = torch.matmul(F.normalize(feature, dim=1, p=2), F.normalize(first_sing_vecs, dim=1, p=2).T)
                    cos_sim = cos_sim.cpu().detach().numpy()
                    pred = cos_sim.argmax(axis=1)[0]
                    sco = cos_sim.max(axis=1)[0]
                else:
                    ret = ret[0]
                    conf = postprocessing(ret, types, **recog_kwarg)
                    y_score.append(conf[0])
                    pred = ret.argmax(axis=1)[0]
                    sco = ret.max(axis=1)[0]
            else:
                conf = postprocessing(ret, types, **recog_kwarg)
                y_score.append(conf[0])
                pred = ret.argmax(axis=1)[0]
                sco = ret.max(axis=1)[0]

            if vis >= 1:
                if args.debug and (idx < 5 or 700 <= idx < 710):  # 调试的时候只展示一部分图片，不然全部都展示很慢
                    pred_label = ret.argmax(axis=1)[0]
                    pred_score = ret.max(axis=1)[0]
                    img = cv2.imread(img_path)
                    cls_name = "neg" if pred_label == len(CLASSES) or pred_label == -1 else CLASSES[pred_label]
                    box_txt = "{}:{:.2f}".format(cls_name, pred_score)
                    img = draw_box_and_kpt(img, gt_box, box_color=PALETTE[gt_label],
                                           box_txt="gt:{}".format(gt_label))
                    img = draw_box_and_kpt(img, gt_box, box_color=PALETTE[pred], box_txt=box_txt)
                    show_img(img)

            if pred == len(CLASSES):
                pred = -1

            if gt_label == -1:
                y_true.append(0)
                if pred == -1:
                    acc_neg += 1
                neg_num += 1
            else:
                y_true.append(1)
                if pred == gt_label:
                    acc_pos += 1
                pos_num += 1
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    print("acc")
    acc_pos_str = str(acc_pos) + " |" + str(pos_num) + " |{:.4f}".format(acc_pos / float(pos_num))
    acc_neg_str = str(acc_neg) + " |" + str(neg_num) + " |{:.4f}".format(acc_neg / float(neg_num))
    print(acc_pos_str)
    print(acc_neg_str)

    auc = roc_auc_score(y_true, y_score)
    print("auc {:.4f}".format(auc))
    if roc_path is not None:
        print("save roc into {}".format(roc_path))
        os.makedirs(os.path.dirname(roc_path), exist_ok=True)
        fw = open(roc_path, "w", encoding="utf-8")
    else:
        fw = None
    roc = get_fprs_with_tprs(y_true, y_score, tprs=[0.95, 0.9, 0.85])
    err_rate_str = ""
    err_rate_str_2 = ""
    recall_str = ""
    recall_str_2 = ""
    for err_rate, recall, precision, threshold in roc:
        res = "err_rate:{} recall:{:.4f} threshold:{:.4f}".format(err_rate, float(recall), threshold)
        print(res)
        if fw is not None:
            fw.write(res + "\n")
        err_rate_str += "{:.4f}".format(err_rate) + "|"
        err_rate_str_2 += "{:.4f}".format(err_rate) + "/"
        recall_str += "{:.4f}".format(recall) + "|"
        recall_str_2 += "{:.4f}".format(recall) + "/"
    print("err_rate markdown")
    print(err_rate_str + " {:.4f}".format(auc) + " |{:.4f}".format(acc_pos / float(pos_num)))
    print(err_rate_str_2 + " {:.4f}".format(auc) + " /{:.4f}".format(acc_pos / float(pos_num)))
    print("recall markdown")
    print(recall_str)
    print(recall_str_2)
    print("roc curve")
    if fw is not None:
        fw.write("err_rate markdown:\n")
        fw.write(err_rate_str + "\n")
        fw.write(err_rate_str_2 + "\n")
        fw.write("recall markdown:\n")
        fw.write(recall_str + "\n")
        fw.write(recall_str_2 + "\n")
        fw.write("auc:\n")
        fw.write("{:.4f}".format(auc) + "\n")
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=None)
    if use_recog_model:
        title_name = config.split("/")[-2]
    else:
        title_name = "curve"
    plot_curve(fpr, tpr, title_name=title_name)
    if fw is not None:
        fw.write("pos acc:\n")
        fw.write(acc_pos_str + "\n")
        fw.write("neg acc:\n")
        fw.write(acc_neg_str + "\n")
        fw.close()
    return gt_box_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--debug', action="store_true", help='debug')
    args = parser.parse_args()
    datas = {
        "cw_neg": {
            "label_path": [
                "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/pred/pred.test.ges30_imgs_1018_1019.map.box.txt",
                "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/neg_list/neg.test.ges30_imgs_1018_1019.map.box.txt",
            ],
            "save_img_root": "/zhoudu/test/gesture/show/yolov3_416_v3/imgs",
            "txt_root": "/zhoudu/test/gesture/show/yolov3_416_v3/cluster_rst/",
            "roc_root": "/zhoudu/test/gesture/roc/yolov3_416_v3/",
            "txt_prefix": "pred",
        },
        "neg_zipai": {
            "label_path": [
                "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/pred/pred.test.ges30_imgs_1018_1019.map.box.txt",
                "/dataset/dataset/ssd/gesture/neg/txt/neg.map.box.txt",
                # "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/pred/neg.test.ges30_imgs_1018_1019.map.box.txt",
                # "/dataset/dataset/ssd/gesture/imagenet/imagenet_ood.box.txt"
            ],
            "save_img_root": "/zhoudu/test/gesture/show/yolov3_416_v3/imgs",
            "txt_root": "/zhoudu/test/gesture/show/yolov3_416_v3/cluster_rst/",
            "roc_root": "/zhoudu/test/gesture/roc/yolov3_416_v3/",
            "txt_prefix": "pred",
        },
        "neg_imagenet": {
            "label_path": [
                "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/pred/pred.test.ges30_imgs_1018_1019.map.box.txt",
                "/dataset/dataset/ssd/gesture/imagenet/imagenet_ood.box.txt"
            ],
            "save_img_root": "/zhoudu/test/gesture/show/yolov3_416_v3/imgs",
            "txt_root": "/zhoudu/test/gesture/show/yolov3_416_v3/cluster_rst/",
            "roc_root": "/zhoudu/test/gesture/roc/yolov3_416_v3/",
            "txt_prefix": "pred",
        },
        "cls7_plus_neg": {
            "label_path": [
                "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/pred/pred.test.ges30_imgs_1018_1019.map.box.txt",
                "/dataset/dataset/ssd/gesture/neg/txt/neg.map.box.txt",
                "/dataset/dataset/ssd/gesture/neg2/txt/neg.map.box.txt",
            ],
            "save_img_root": "/zhoudu/test/gesture/show/yolov3_416_v3/imgs",
            "txt_root": "/zhoudu/test/gesture/show/yolov3_416_v3/cluster_rst/",
            "roc_root": "/zhoudu/test/gesture/roc/yolov3_416_v3/",
            "txt_prefix": "pred",
        },
        "cls7_indoor_multi_scenario": {
            "label_path": [
                "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/pred/pred.test.indoor_multi_scenario.map.box.txt",
                "/dataset/dataset/ssd/gesture/neg/txt/neg.map.box.txt",
                "/dataset/dataset/ssd/gesture/neg2/txt/neg.map.box.txt",
            ],
            "save_img_root": "/zhoudu/test/gesture/show/yolov3_416_v3/imgs",
            "txt_root": "/zhoudu/test/gesture/show/yolov3_416_v3/cluster_rst/",
            "roc_root": "/zhoudu/test/gesture/roc/yolov3_416_v3/",
            "txt_prefix": "pred",
        }
    }

    recog_kwarg = {
        "config": "tools/gesture_recog/exp_rodd_cls7/config.py",
        "model": "tools.gesture_recog.exp_rodd_cls7.model.Model",
        "weight_path": "/zhoudu/checkpoints/gesture_recog/exp_rodd_cls7/checkpoint/model-epoch189.weights",
        "first_sing_vec_path": "/zhoudu/test/gesture/feas/exp_rodd_cls7/first_sing_vec.train.ges30_cloudwalk.ges30_imgs_1015_1017.npy"
    }
    vis = 0  # 0: 不可视化， 1：存储为图片 2：plt展示
    use_recog_model = True
    types = "rodd"  # msp proser rodd arcface
    sco_thd = 0.0
    offset = 0
    if args.debug:
        vis = 2
    # data_name = ["cw_neg", "neg_zipai", "neg_imagenet"]
    data_name = ["cls7_indoor_multi_scenario"]
    for data_name in data_name:
        print("data name:{}".format(data_name))
        if use_recog_model:
            from tools.gesture_recog.build_model import build_model, get_gesture_inputs, get_test_transformer

            weight_path_sp = recog_kwarg["weight_path"].split("/")
            roc_path_name = os.path.join((weight_path_sp[-3]), weight_path_sp[-1], data_name + "_result.txt")
        else:
            roc_path_name = os.path.join("origin", data_name + "_result.txt")

        label_path = datas[data_name]["label_path"]
        save_img_root = datas[data_name]["save_img_root"]
        txt_root = datas[data_name]["txt_root"]
        txt_prefix = datas[data_name]["txt_prefix"]
        roc_root = datas[data_name]["roc_root"]
        if isinstance(label_path, list):
            root, name = os.path.split(label_path[0])
            label_path_t = os.path.join(root, "_merge." + name)
            get_merge_txt_seq(label_path, label_path_t)
            label_path = label_path_t
        print("EVAL:{}".format(label_path))
        roc_path = os.path.join(roc_root, roc_path_name)
        eval(label_path, vis=vis, offset=offset,
             sco_thd=sco_thd, img_root=save_img_root, txt_root=txt_root,
             txt_prefix=txt_prefix, use_recog_model=use_recog_model, roc_path=roc_path,
             types=types, **recog_kwarg)
        gt_num = len(open(label_path, "r", encoding="utf-8").readlines())
        offset += gt_num
