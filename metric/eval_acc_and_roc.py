# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 14:04
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : eval.py
# @Software: PyCharm
import os
import sys
import argparse
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.box_and_kpt_utils import IOU_XYWH, get_area
from metric.write_info import write_acc_and_roc, save_result


# os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def eval_acc_and_roc(gt_path, pred_path, roc_path=None, title_name=None):
    gts = defaultdict(list)
    gt_pos_box_num = 0
    gt_pos_img = set()
    gt_neg_box_num = 0
    gt_neg_img = set()
    with open(gt_path, "r", encoding="utf-8") as fr:
        for idx, line in enumerate(fr.readlines()):
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            label = int(line_sp[1])
            if label >= 0:
                gt_pos_box_num += 1
                gt_pos_img.add(img_path)
            else:
                gt_neg_box_num += 1
                gt_neg_img.add(img_path)
            box = list(map(float, line_sp[4:8]))  # x, y, w, h
            img_w, img_h = int(line_sp[2]), int(line_sp[3])
            data = {"img_path": img_path, "label": label, "box": box, "img_w": img_w, "img_h": img_h, "line": line}
            gts[img_path].append(data)
    print("gt_pos_box_num:{} gt_neg_box_num:{}".format(gt_pos_box_num, gt_neg_box_num))
    preds = defaultdict(list)
    with open(pred_path, "r", encoding="utf-8") as fr:
        for idx, line in enumerate(fr.readlines()):
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            label = int(line_sp[1])
            box = list(map(float, line_sp[4:8]))  # x, y, w, h
            img_w, img_h = int(line_sp[2]), int(line_sp[3])
            sco = float(line_sp[8])  # 该类别的分数
            conf = float(line_sp[9])  # 一般用于判断是否是ood数据
            data = {"img_path": img_path, "label": label, "box": box, "img_w": img_w, "img_h": img_h, "sco": sco,
                    "conf": conf, "line": line}
            preds[img_path].append(data)

    y_true = []  # 记录对应识别框的手是否是ID数据，1表示id数据，0表示OOD数据
    y_score = []
    acc_pos = 0
    pos_num = 0
    acc_neg = 0
    neg_num = 0
    miss_box_num = 0
    false_det_num = 0
    result_info = defaultdict(list)
    for idx, img_path in enumerate(gts.keys()):
        if "WIN_20220715_11_32_54_Pro_000005800" in img_path:
            print()
        gt = gts[img_path]
        if img_path not in preds:  # 预测结果中没有该文件，有两种情况，如果是正例没有预测出来则是错误的，如果是负例没有预测出来则是正确的
            if img_path in gt_pos_img:
                pos_num += len(gt)
                miss_box_num += 1
                for i in range(len(gt)):
                    result_info["miss"].append([-1, gt[i]["line"]])
                continue
            elif img_path in gt_neg_img:
                neg_num += len(gt)
                acc_neg += 1
                continue
        pred = preds[img_path]
        # 获取误检的性能
        false_det_dict = {}
        max_iou = -1
        for pred_idx, pred_data in enumerate(pred):
            pred_box = pred_data["box"]
            is_false_det = True
            for gt_idx, gt_data in enumerate(gt):
                gt_box = gt_data["box"]
                gt_box_str = " ".join(list(map(lambda x: "{:.2f}".format(x), gt_box)))
                iou = IOU_XYWH(pred_box, gt_box)
                if gt_box_str not in false_det_dict:
                    false_det_dict[gt_box_str] = 0
                if iou > 0.0:
                    false_det_dict[gt_box_str] += 1

                    is_false_det = False

            if is_false_det:  # 如果一个框都没有匹配上，则一定是误检
                false_det_num += 1
                result_info["false_det"].append([-1, pred_data["line"]])
        for gt_box_str in false_det_dict:
            if false_det_dict[gt_box_str] >= 2:
                false_det_num += max(false_det_dict[gt_box_str] - 1, 0)  # 这些数据没有记录上误检result_info["false_det"]

        # 获取识别的性能
        for gt_data in gt:
            gt_box = gt_data["box"]
            gt_label = gt_data["label"]
            max_area = -1
            max_iou = -1
            max_iou_index = -1
            reserve_data = None
            false_det_index = []
            for pred_idx, pred_data in enumerate(pred):
                pred_box = pred_data["box"]
                pred_label = pred_data["label"]
                iou = IOU_XYWH(pred_box, gt_box)
                if iou > 0.0:
                    # area = get_area(pred_box)
                    # if area > max_area:
                    #     max_area = area
                    #     reserve_data = pred_data
                    false_det_index.append(pred_idx)
                    if iou > max_iou:
                        max_iou = iou
                        max_iou_index = pred_idx
                        reserve_data = pred_data
            if max_iou_index != -1:
                false_det_index.remove(max_iou_index)
            if reserve_data is None:
                miss_box_num += 1
                pos_num += 1
                result_info["miss"].append([gt_label, gt_data["line"]])
            else:
                # 记录误检的情况
                for pred_idx in false_det_index:
                    pred_data = pred[pred_idx]
                    result_info["false_det"].append([0, pred_data["line"]])

                pred_label = reserve_data["label"]
                conf = reserve_data["conf"]
                y_score.append(conf)
                if gt_label == -1:
                    y_true.append(0)
                    if pred == -1:
                        acc_neg += 1
                    neg_num += 1
                else:
                    y_true.append(1)
                    if pred_label == gt_label:
                        acc_pos += 1
                        result_info["right"].append([gt_label, reserve_data["line"]])
                    else:
                        result_info["wrong"].append([gt_label, reserve_data["line"]])
                    pos_num += 1
    print("miss_box_num", miss_box_num)
    print("pos_num", pos_num)
    print("false_det_num", false_det_num)
    write_acc_and_roc(acc_neg, acc_pos, neg_num, pos_num, len(gt_pos_img), roc_path, title_name, y_score, y_true,
                      miss_box_num,
                      false_det_num)
    return result_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--debug', action="store_true", help='debug')
    parser.add_argument('--gt_path',
                        default="/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/test.ges30_imgs_1018_1019.map.box.txt",
                        type=str, help='the pred path')
    parser.add_argument('--pred_path',
                        default="/zhoudu/test/gesture/recog/exp_rodd_cls7_model-epoch189.weights/test_ges30_imgs_1018_1019_cls7/recog.txt",
                        type=str, help='the pred path')
    parser.add_argument('--save_roc_path',
                        default="/zhoudu/test/gesture/recog/exp_rodd_cls7_model-epoch189.weights/test_ges30_imgs_1018_1019_cls7/roc.txt",
                        type=str, help='save roc path')
    parser.add_argument('--title_name', default=None, help='the pred path')
    parser.add_argument('--vis', default=False, help='the pred path')
    parser.add_argument('--txt_root', default="", help='if vis is True, the result will save into this dir')
    parser.add_argument('--txt_prefix', default="", help='if vis is True, the file prefix')
    parser.add_argument('--src_img_prefix', default="/dataset/dataset/ssd/gesture/",
                        help='if vis is True, the source img path prefix')
    parser.add_argument('--save_img_root', default="/zhoudu/golabel/demo/imgs",
                        help='if vis is True, the vis img will save into this dir')

    args = parser.parse_args()
    pred_path = args.pred_path
    gt_path = args.gt_path
    roc_path = args.save_roc_path
    title_name = args.title_name
    result_info = eval_acc_and_roc(gt_path, pred_path, roc_path, title_name=title_name)
    if args.vis:
        txt_root = args.txt_root
        txt_prefix = args.txt_prefix
        src_img_prefix = args.src_img_prefix
        save_img_root = args.save_img_root
        save_result(result_info, txt_root, txt_prefix, src_img_prefix, save_img_root)
