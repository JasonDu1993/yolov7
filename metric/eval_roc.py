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
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.box_and_kpt_utils import IOU_XYWH
from utils.draw_box_kpt_utils import draw_box_and_kpt, show_img, save_img
import matplotlib.pyplot as plt
from utils.datas.gesture.class_color_c9 import CLASSES, PALETTE
from metric.roc import get_roc_with_err_rates, get_fprs_with_tprs
from utils.plot_curve import plot_curve
from utils.datas.gesture.merge import get_merge_txt_seq
from utils.box_and_kpt_utils import xyxy2xywh, get_area


def eval(result_path, label_path, vis=0, offset=0, sco_thd=0.5, img_root="/zhoudu/test/gesture/show/yolov3_416_23/imgs",
         txt_root="/zhoudu/test/gesture/show/yolov3_416_23/cluster_rst/", txt_prefix="", use_recog_model=False,
         use_gt_box=False, roc_path=None, **kwargs):
    if not use_gt_box:
        results = mmcv.load(result_path)  # list[list[ndarray,...]
    else:
        results = []
    gts = defaultdict(list)
    gt_box_num = 0
    with open(label_path, "r", encoding="utf-8") as fr:
        for idx, line in enumerate(fr.readlines()):
            res = []
            gt_box_num += 1
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            label = int(line_sp[1])
            box = list(map(float, line_sp[4:8]))  # x, y, w, h
            # if vis >= 1:
            #     img = cv2.imread(img_path)
            #     img = draw_box_and_kpt(img, box)
            #     show_img(img)
            if use_gt_box:
                x, y, w, h = box
                res.append(np.array([x, y, x + w, y + h, 1]).reshape(1, 5))
                results.append(res)
            img_w, img_h = int(line_sp[2]), int(line_sp[3])
            data = {"img_path": img_path, "label": label, "box": box, "img_w": img_w, "img_h": img_h}
            gts[idx].append(data)  # 一般一个idx对应一个gt框
    results = results[offset:offset + gt_box_num]
    preds = defaultdict(list)
    if use_recog_model:
        config = kwargs["config"]
        model = kwargs["model"]
        weight_path = kwargs["weight_path"]
        gesture_model = build_model(config, model, weight_path)
        gpu = 0
        gesture_model.to("cuda:{}".format(gpu))
        trans = get_test_transformer(config)
    cnt = 0
    cnt_total = 0
    y_true = []
    y_score = []
    acc_pos = 0
    pos_num = 0
    acc_neg = 0
    neg_num = 0
    for idx in range(len(results)):
        if idx == 700:
            print("dd")
        result = results[idx]
        gt = gts[idx]
        gt_data = gt[0]  # 假设一张图只有一个gt框
        img_path = gt_data["img_path"]
        gt_label = gt_data["label"]
        gt_box = gt_data["box"]
        img = cv2.imread(img_path)

        max_area = -1
        max_area_cnt = -1
        reserve_data = None
        reserve_conf = None
        for label in range(len(result)):
            bboxes = result[label]  # ndarray
            for i in range(bboxes.shape[0]):
                data = dict()
                bbox = xyxy2xywh(bboxes[i])
                score = float(bboxes[i][4])
                data['bbox'] = bbox
                data['score'] = score
                data['label'] = label
                cnt_total += 1
                if use_recog_model:
                    kpts = None
                    img_new, bbox_new, kpts, img_path = trans(img, bbox, kpts, img_path)
                    img_new = img_new.unsqueeze(0)
                    img_new = img_new.to(gpu)
                    # 使用proser方法的模型进行测试
                    ret = gesture_model(img_new, mode="test")
                    ret = F.softmax(ret, dim=1)
                    ret = ret.cpu().detach().numpy()
                    dummyconf = ret[:, -1]
                    maxknownconf = np.max(ret[:, :-1], axis=1)
                    conf = maxknownconf - dummyconf

                    # 使用原始的方法进行训练的模型进行测试
                    # ret = gesture_model(img_new)
                    # ret = F.softmax(ret, dim=1)
                    # ret = ret.cpu().detach().numpy()
                    # conf = np.max(ret, axis=1)

                    pred = np.argmax(ret, axis=1)
                    # if gt_label == -1:
                    #     if pred[0] == 14:
                    #         acc_neg += 1
                    #     neg_num += 1
                    # else:
                    #     if pred[0] == gt_label:
                    #         acc_pos += 1
                    #     pos_num += 1

                    if vis >= 1:
                        if args.debug and (idx < 5 or 700 <= idx < 710):  # 调试的时候只展示一部分图片，不然全部都展示很慢
                            pred_label = ret.argmax(axis=1)[0]
                            pred_score = ret.max(axis=1)[0]
                            iou = IOU_XYWH(bbox, gt_box)
                            img = cv2.imread(img_path)
                            cls_name = "neg" if pred_label == len(CLASSES) or pred_label == -1 else CLASSES[pred_label]
                            box_txt = "{}:{:.2f}:{:.2f}".format(cls_name, pred_score, iou)
                            img = draw_box_and_kpt(img, gt_box, box_color=PALETTE[gt_label],
                                                   box_txt="gt:{}".format(gt_label))
                            img = draw_box_and_kpt(img, bbox, box_color=PALETTE[pred[0]], box_txt=box_txt)
                            show_img(img)
                    pred = ret.argmax(axis=1)[0]
                    sco = ret.max(axis=1)[0]
                    # if label != pred:
                    #     if score < sco:
                    cnt += 1
                    if pred == len(CLASSES):
                        pred = -1
                    data['score'] = sco
                    data['label'] = pred
                area = get_area(bbox)
                if area > max_area:
                    max_area = area
                    reserve_data = data
                    reserve_conf = conf
                    max_area_cnt += 1
                    if max_area_cnt >= 1:
                        print(idx, img_path)
        if reserve_data is not None:
            if gt_label == -1:
                y_true.append(0)
                if reserve_data["label"] == -1:
                    acc_neg += 1
                neg_num += 1

            else:
                y_true.append(1)
                if reserve_data["label"] == gt_label:
                    acc_pos += 1
                pos_num += 1
            y_score.append(reserve_conf[0])
            preds[idx].append(reserve_data)
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
    roc = get_fprs_with_tprs(y_true, y_score)
    err_rate_str = ""
    recall_str = ""
    for err_rate, recall, precision, threshold in roc:
        res = "err_rate:{} recall:{:.4f} threshold:{:.4f}".format(err_rate, float(recall), threshold)
        print(res)
        if fw is not None:
            fw.write(res + "\n")
        err_rate_str += "{:.4f}".format(err_rate) + "|"
        recall_str += "{:.4f}".format(recall) + "|"
    print("err_rate markdown")
    print(err_rate_str + " {:.4f}".format(auc))
    print("recall markdown")
    print(recall_str)
    print("roc curve")
    if fw is not None:
        fw.write("err_rate markdown:\n")
        fw.write(err_rate_str + "\n")
        fw.write("recall markdown:\n")
        fw.write(recall_str + "\n")
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
    # result_path = "/zhoudu/test/gesture/metric/yolo_for_gesture/yolov3_416_v3/epoch_270.pth/out.pkl"
    # result_path = "/zhoudu/test/gesture/metric/yolo_for_gesture/yolov3_416_v3_mobilenetv2/epoch_270.pth/out_o.pkl"
    result_paths = [
        [
            "/zhoudu/test/gesture/metric/yolo_for_gesture/yolov3_416_v3_mobilenetv2/epoch_270.pth/test.ges30_imgs_1018_1019.map.box.pkl",
            "/dataset/dataset/ssd/gesture/neg/txt/neg.pkl"
        ],
    ]
    label_paths = [
        [
            "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/merge.test.ges30_imgs_1018_1019.map.box.txt",
            "/dataset/dataset/ssd/gesture/neg/txt/neg.map.box.txt",
        ]
    ]
    # label_path = "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/neg.test.ges30_imgs_1018_1019.map.box.txt"
    save_img_roots = [
        "/zhoudu/test/gesture/show/yolov3_416_v3/imgs",
    ]
    txt_roots = [
        "/zhoudu/test/gesture/show/yolov3_416_v3/cluster_rst/",
    ]
    roc_roots = [
        "/zhoudu/test/gesture/roc/yolov3_416_v3/",
    ]
    txt_prefixs = [
        "merge",
    ]
    vis = 0  # 0: 不可视化， 1：存储为图片 2：plt展示
    if args.debug:
        vis = 2
    use_recog_model = True
    use_gt_box = False
    sco_thd = 0.0
    recog_kwarg = {}
    if use_recog_model:
        from tools.gesture_recog.build_model import build_model, get_gesture_inputs, get_test_transformer

        config = "tools/gesture_recog/exp220628a/config.py"
        model = "tools.gesture_recog.exp220628a.model.Model"
        weight_path = "/zhoudu/checkpoints/gesture_recog/gesture220628a_113217/checkpoint/model-epoch179.weights"
        recog_kwarg["config"] = config
        recog_kwarg["model"] = model
        recog_kwarg["weight_path"] = weight_path
        weight_path_sp = weight_path.split("/")
        roc_path_name = os.path.join((weight_path_sp[-3]), weight_path_sp[-1], "result.txt")
    else:
        roc_path_name = os.path.join("origin", "result.txt")
    offset = 0
    cnt = 0
    for result_path, label_path, save_img_root, txt_root, txt_prefix, roc_root in \
            zip(result_paths, label_paths, save_img_roots, txt_roots, txt_prefixs, roc_roots):
        if isinstance(label_path, list):
            root, name = os.path.split(label_path[0])
            label_path_t = os.path.join(root, "merge." + name)
            get_merge_txt_seq(label_path, label_path_t)
            label_path = label_path_t
        if isinstance(result_path, list):
            res = []
            for r in result_path:
                out = mmcv.load(r)
                res.extend(out)
            root, name = os.path.split(result_path[0])
            result_path = os.path.join(root, "merge." + name)
            mmcv.dump(res, result_path)
        print("EVAL:{}".format(label_path))
        cnt += 1
        # if cnt == 2:
        roc_path = os.path.join(roc_root, roc_path_name)
        eval(result_path, label_path, vis=vis, offset=offset,
             sco_thd=sco_thd, img_root=save_img_root, txt_root=txt_root,
             txt_prefix=txt_prefix, use_recog_model=use_recog_model, use_gt_box=use_gt_box, roc_path=roc_path,
             **recog_kwarg)
        gt_num = len(open(label_path, "r", encoding="utf-8").readlines())
        offset += gt_num
