# -*- coding: utf-8 -*-
# @Time    : 2022/7/8 11:50
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : write_info.py
# @Software: PyCharm
import os
import numpy as np
import shutil

from metric.roc import get_fprs_with_tprs
from utils.plot_curve import plot_curve
from sklearn.metrics import roc_auc_score, roc_curve
from vis.gen_golabel import gen_golabel_from_txt
from utils.datas.gesture.class_color_c9 import CLASSES, PALETTE


def write_acc_and_roc(acc_neg, acc_pos, neg_num, pos_num, gt_pos_img_num, roc_path, title_name, y_score, y_true,
                      miss_box_num,
                      false_det_num):
    if roc_path is not None:
        print("save roc into {}".format(roc_path))
        os.makedirs(os.path.dirname(roc_path), exist_ok=True)
        fw = open(roc_path, "w", encoding="utf-8")
    else:
        fw = None
    # 统计正例的准确率
    print("pos acc:")
    acc_pos_str = str(acc_pos) + " |" + str(pos_num) + " |{:.4f}".format(acc_pos / float(pos_num))
    print(acc_pos_str)
    if fw is not None:
        fw.write("pos acc:\n")
        fw.write(acc_pos_str + "\n")

    # 统计误检率
    print("false det:")
    false_det_str = str(false_det_num) + " |" + str(gt_pos_img_num) + " |{:.4f}".format(
        false_det_num / float(gt_pos_img_num))
    print(false_det_str)
    if fw is not None:
        fw.write("false_det:\n")
        fw.write(false_det_str + "\n")
    # 统计漏检率
    print("miss:")
    miss_pos_str = str(miss_box_num) + " |" + str(gt_pos_img_num) + " |{:.4f}".format(
        miss_box_num / float(gt_pos_img_num))
    print(miss_pos_str)
    if fw is not None:
        fw.write("miss:\n")
        fw.write(miss_pos_str + "\n")
    print("acc|误检|漏检")
    print("{:.4f}".format(acc_pos / float(pos_num)) +
          " /{:.4f}".format(false_det_num / float(gt_pos_img_num)) +
          " /{:.4f}".format(miss_box_num / float(gt_pos_img_num))
          )
    if neg_num > 0:
        # 统计负例的准确率
        acc_neg_str = str(acc_neg) + " |" + str(neg_num) + " |{:.4f}".format(acc_neg / float(neg_num))
        print(acc_neg_str)
        if fw is not None:
            fw.write("neg acc:\n")
            fw.write(acc_neg_str + "\n")
        # 统计auc
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        auc = roc_auc_score(y_true, y_score)
        print("auc {:.4f}".format(auc))
        if fw is not None:
            fw.write("auc:\n")
            fw.write("{:.4f}".format(auc) + "\n")
        # 统计roc
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
        if fw is not None:
            fw.write("err_rate markdown:\n")
            fw.write(err_rate_str + "\n")
            fw.write(err_rate_str_2 + "\n")
        print("recall markdown")
        print(recall_str)
        print(recall_str_2)
        if fw is not None:
            fw.write("recall markdown:\n")
            fw.write(recall_str + "\n")
            fw.write(recall_str_2 + "\n")

        if title_name is not None:
            print("roc curve")
            fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=None)
            plot_curve(fpr, tpr, title_name=title_name)
    if fw is not None:
        fw.close()


def save_result(result_info, txt_root, txt_prefix, src_img_prefix, save_img_root, vis=True, save_right=False,
                resize=False, dst_size=(416, 416)):
    """

    Args:
        result_info: dict, key is "right", "wrong"
    """
    os.makedirs(txt_root, exist_ok=True)
    wrong_path = os.path.join(txt_root, txt_prefix + "wrong.txt")
    print("wrong_path: {}".format(wrong_path))
    with open(wrong_path, "w", encoding="utf-8") as fw:
        for gt_label, line in result_info["wrong"]:
            new_line = line.strip() + " " + str(gt_label) + "\n"
            fw.write(new_line)
    if save_right:
        right_path = os.path.join(txt_root, txt_prefix + "right.txt")
        print("right_path: {}".format(right_path))
        with open(right_path, "w", encoding="utf-8") as fw:
            for gt_label, line in result_info["right"]:
                fw.write(line)

    if vis:
        save_path = wrong_path + ".t"
        gen_golabel_from_txt(wrong_path, save_path, vis=vis, is_contain_imgwh=True, src_img_prefix=src_img_prefix,
                             save_img_root=save_img_root, use_gt_label=True, resize=resize, dst_size=dst_size)
        shutil.move(save_path, wrong_path)

        if save_right:
            save_path = right_path + ".t"
            gen_golabel_from_txt(right_path, save_path, vis=vis, is_contain_imgwh=True, src_img_prefix=src_img_prefix,
                                 save_img_root=save_img_root, resize=resize, dst_size=dst_size)
            shutil.move(save_path, right_path)


def parse_error(result_info):
    gt_pred = {}
    for gt_label, line in result_info["wrong"] + result_info["right"]:
        gt_label = int(gt_label)
        line_sp = line.strip().split(" ")
        pred_label = int(line_sp[1])
        if gt_label not in gt_pred:
            gt_pred[gt_label] = {"num": 0, "error_num": 0, "right_num": 0}
        gt_pred[gt_label]["num"] += 1
        if gt_label == pred_label:
            gt_pred[gt_label]["right_num"] += 1
        else:
            if pred_label not in gt_pred[gt_label]:
                gt_pred[gt_label][pred_label] = 0
            gt_pred[gt_label][pred_label] += 1
            gt_pred[gt_label]["error_num"] += 1

    for gt_label in sorted(gt_pred.keys()):
        gt_dict = gt_pred[gt_label]
        gt_num = gt_dict["num"]
        gt_right_num = gt_dict["right_num"]
        gt_error_num = gt_dict["error_num"]
        acc = gt_right_num / gt_num
        errs = {}
        for x in gt_dict:
            if isinstance(x, int):
                n = gt_dict[x]
                err = n / gt_error_num
                errs[x] = err
        print("gt:{} acc:{:.2f} right:{} gt:{}".format(CLASSES[int(gt_label)], acc, gt_right_num, gt_num))
        for x in errs:
            print("gt:{} pred:{} err:{:.2f}".format(CLASSES[int(gt_label)], CLASSES[int(x)], errs[x]))
        print()


def save_false_det(result_info, txt_root, txt_prefix, src_img_prefix, save_img_root, vis=True):
    """

    Args:
        result_info: dict, key is "right", "wrong"
    """
    if "false_det" not in result_info:
        return
    os.makedirs(txt_root, exist_ok=True)
    false_det_path = os.path.join(txt_root, txt_prefix + "false_det.txt")
    print("false_det_path: {}".format(false_det_path))
    with open(false_det_path, "w", encoding="utf-8") as fw:
        for gt_label, line in result_info["false_det"]:
            new_line = line.strip() + " " + str(gt_label) + "\n"
            fw.write(new_line)
    if vis:
        save_path = false_det_path + ".t"
        gen_golabel_from_txt(false_det_path, save_path, vis=vis, is_contain_imgwh=True, src_img_prefix=src_img_prefix,
                             save_img_root=save_img_root, use_gt_label=True)
        shutil.move(save_path, false_det_path)


def save_miss_det(result_info, txt_root, txt_prefix, src_img_prefix, save_img_root, vis=True):
    """

    Args:
        result_info: dict, key is "right", "wrong"
    """
    if "miss" not in result_info:
        return
    os.makedirs(txt_root, exist_ok=True)
    save_path = os.path.join(txt_root, txt_prefix + "miss.txt")
    print("miss_path: {}".format(save_path))
    with open(save_path, "w", encoding="utf-8") as fw:
        for gt_label, line in result_info["miss"]:
            new_line = line.strip() + " " + str(gt_label) + "\n"
            fw.write(new_line)
    if vis:
        vis_save_path = save_path + ".t"
        gen_golabel_from_txt(save_path, vis_save_path, vis=vis, is_contain_imgwh=True, src_img_prefix=src_img_prefix,
                             save_img_root=save_img_root, use_gt_label=True)
        shutil.move(vis_save_path, save_path)
