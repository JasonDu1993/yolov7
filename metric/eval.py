# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 14:04
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : eval.py
# @Software: PyCharm
import os
import sys
import mmcv
import cv2
from collections import defaultdict
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.box_and_kpt_utils import IOU_XYWH
from utils.draw_box_kpt_utils import draw_box_and_kpt, show_img, save_img
from utils.datas.gesture.class_color_c9 import CLASSES, PALETTE
from utils.box_and_kpt_utils import xyxy2xywh, get_area


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def eval(result_path, label_path, vis=0, offset=0, sco_thd=0.5, img_root="/zhoudu/test/gesture/show/yolov3_416_23/imgs",
         txt_root="/zhoudu/test/gesture/show/yolov3_416_23/cluster_rst/", txt_prefix="", use_recog_model=False,
         **kwargs):
    results = mmcv.load(result_path)  # list[list[ndarray,...]
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
            gts[idx].append(data)  # 一般一个idx对应一个gt框
    results = results[offset:offset + gt_box_num]
    preds = defaultdict(list)
    if use_recog_model:
        config = kwargs["config"]
        model = kwargs["model"]
        weight_path = kwargs["weight_path"]
        gesture_model = build_model(config, model, weight_path)
        gpu = 3
        gesture_model.to("cuda:{}".format(gpu))
        trans = get_test_transformer(config)
    cnt = 0
    cnt_total = 0
    for idx in range(len(results)):
        result = results[idx]
        gt = gts[idx]
        gt_data = gt[0]  # 假设一张图只有一个gt框
        img_path = gt_data["img_path"]
        gt_label = gt_data["label"]
        gt_box = gt_data["box"]
        img = cv2.imread(img_path)

        max_area = -1
        reserve_data = None
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
                    ret = gesture_model(img_new, mode="test")
                    if isinstance(ret, tuple):
                        ret = ret[0]
                    ret = F.softmax(ret, dim=1)
                    ret = ret.cpu().detach().numpy()
                    pred = ret.argmax(axis=1)[0]
                    sco = ret.max(axis=1)[0]
                    # if label != pred:
                    #     if score < sco:
                    cnt += 1
                    if pred == len(CLASSES):
                        pred = -1
                    data['score'] = sco
                    data['label'] = pred
                    if vis == 2:
                        pred_label = ret.argmax(axis=1)[0]
                        pred_score = ret.max(axis=1)[0]
                        iou = IOU_XYWH(bbox, gt_box)
                        img = cv2.imread(img_path)
                        box_txt = "{}:{:.2f}:{:.2f}".format(CLASSES[pred_label], pred_score, iou)
                        img = draw_box_and_kpt(img, gt_box, box_color=PALETTE[gt_label],
                                               box_txt="gt:{}".format(gt_label))
                        img = draw_box_and_kpt(img, bbox, box_color=PALETTE[pred], box_txt=box_txt)
                        show_img(img)
                area = get_area(bbox)
                if area > max_area:
                    max_area = area
                    reserve_data = data
        preds[idx].append(reserve_data)
    print("modify {}/{}, {:.2f}".format(cnt, cnt_total, cnt / float(cnt_total)))
    cnt_gt = 0  # 记录有多少预测框和gt框的iou大于0的个数
    cnt_true = 0  # 记录有多少预测框和gt框的iou大于等于0.5，并且预测的类别正确，同时分数也大于0.5的框个数
    cnt_none = 0  # 记录有多少预测框和gt框的iou大于等于0.5，并且预测的类别正确，同时分数也大于0.5的框个数
    cnt_wrong = 0  # 记录有多少错误的预测
    cnt_wrong_box_miss = 0  # 记录有多少gt框没有预测出来
    cnt_wrong_box_off = 0  # 记录有多少预测框和gt框的iou小于0.5
    cnt_wrong_class = 0  # 记录有多少预测框和gt框的iou大于等于0.5，但是预测的标签错误
    cnt_wrong_class_sco = 0  # 记录有多少预测框和gt框的iou大于等于0.5，标签正确，但是类别分数小于0.5
    neg_total_num = 0  # 记录检测模型最后保留的负样本个数

    # 使用golabel展示的文件
    if vis >= 1:
        os.makedirs(txt_root, exist_ok=True)
        wrong_path = os.path.join(txt_root, txt_prefix + "wrong.txt")
        print("wrong_path: {}".format(wrong_path))
        wrong_box_miss_path = os.path.join(txt_root, txt_prefix + "wrong_box_miss.txt")
        wrong_box_off_path = os.path.join(txt_root, txt_prefix + "wrong_box_off.txt")
        wrong_class_path = os.path.join(txt_root, txt_prefix + "wrong_class.txt")
        wrong_class_sco_path = os.path.join(txt_root, txt_prefix + "wrong_class_sco.txt")
        right_path = os.path.join(txt_root, txt_prefix + "right.txt")
        f_wrong = open(wrong_path, "w", encoding="utf-8")
        f_wrong_box_miss = open(wrong_box_miss_path, "w", encoding="utf-8")
        f_wrong_box_off = open(wrong_box_off_path, "w", encoding="utf-8")
        f_wrong_class = open(wrong_class_path, "w", encoding="utf-8")
        f_wrong_class_sco = open(wrong_class_sco_path, "w", encoding="utf-8")
        f_right = open(right_path, "w", encoding="utf-8")
    l = len("/dataset/dataset/ssd/gesture/")
    for idx in range(len(results)):
        if idx % 100 == 0:
            print("deal {} image".format(idx))
        gt = gts[idx]
        pred = preds[idx]
        for gt_data in gt:
            img_path = gt_data["img_path"]
            gt_label = gt_data["label"]
            gt_box = gt_data["box"]  # x, y, w, h
            img_w = gt_data["img_w"]
            img_h = gt_data["img_h"]

            cnt_pred = 0  # 记录预测的框中有多少个是和gt框匹配的，只要iou大于0就匹配
            if gt_label == -1 and pred[0] is None:
                cnt_none += 1
                cnt_true += 1
                continue  # 对于负例只统计到这里
            for pred_data in pred:
                pred_bbox = pred_data['bbox']
                pred_score = pred_data['score']
                pred_label = pred_data['label']
                iou = IOU_XYWH(pred_bbox, gt_box)
                if vis >= 1:
                    img = cv2.imread(img_path)
                    # img = draw_box_and_kpt(img, gt_box, box_color=PALETTE[gt_label], box_txt="gt:" + CLASSES[gt_label])
                box_txt = "{}:{:.2f}:{:.2f}".format(CLASSES[pred_label], pred_score, iou)
                if gt_label == -1:
                    neg_total_num += 1
                    if gt_label == pred_label and pred_score >= sco_thd:
                        cnt_true += 1
                    else:
                        cnt_wrong += 1
                        cnt_wrong_class += 1
                        if vis >= 1:
                            img = draw_box_and_kpt(img, gt_box, box_color=PALETTE[gt_label],
                                                   box_txt="gt:-1")
                            img = draw_box_and_kpt(img, pred_bbox, box_color=PALETTE[pred_label], box_txt=box_txt)
                            new_img_path = os.path.join(img_root, "wrong_box_neg", img_path[l:])
                            if vis == 1:
                                save_img(new_img_path, img)
                                line = new_img_path + " " + str(gt_label) + " 1\n"
                                f_wrong_class.write(line)
                            elif vis == 2:
                                show_img(img)

                    continue  # 对于负例只统计到这里
                if iou <= 0:
                    continue
                else:
                    cnt_pred += 1

                if 0 < iou < 0.5:
                    cnt_wrong_box_off += 1
                    cnt_wrong += 1
                    if vis >= 1:
                        img = draw_box_and_kpt(img, pred_bbox, box_color=PALETTE[pred_label], box_txt=box_txt)
                        new_img_path = os.path.join(img_root, "wrong_box_off", img_path[l:])
                        if vis == 1:
                            save_img(new_img_path, img)
                            line = new_img_path + " " + str(gt_label) + " 1\n"
                            f_wrong_box_off.write(line)
                        elif vis == 2:
                            show_img(img)
                    continue
                elif iou >= 0.5:
                    if gt_label != pred_label:
                        cnt_wrong_class += 1
                        cnt_wrong += 1
                        if vis >= 1:
                            img = draw_box_and_kpt(img, pred_bbox, box_color=PALETTE[pred_label], box_txt=box_txt)
                            if vis == 1:
                                new_img_path = os.path.join(img_root, "wrong_class", img_path[l:])
                                save_img(new_img_path, img)
                                line = new_img_path + " " + str(gt_label) + " 2\n"
                                f_wrong_class.write(line)
                            elif vis == 2:
                                show_img(img)
                    elif pred_score < sco_thd:
                        cnt_wrong_class_sco += 1
                        cnt_wrong += 1
                        if vis >= 1:
                            if vis == 1:
                                img = draw_box_and_kpt(img, pred_bbox, box_color=PALETTE[pred_label], box_txt=box_txt)
                                new_img_path = os.path.join(img_root, "wrong_class_sco", img_path[l:])
                                save_img(new_img_path, img)
                                line = new_img_path + " " + str(gt_label) + " 3\n"
                                f_wrong_class_sco.write(line)
                            elif vis == 2:
                                show_img(img)
                    else:
                        cnt_true += 1
                        if vis >= 1:
                            if vis == 1:
                                img = draw_box_and_kpt(img, pred_bbox, box_color=PALETTE[pred_label], box_txt=box_txt)
                                new_img_path = os.path.join(img_root, "right", img_path[l:])
                                save_img(new_img_path, img)
                                line = new_img_path + " " + str(gt_label) + " 4\n"
                                # print(line)
                                f_right.write(line)
                                f_right.flush()
                            elif vis == 2:
                                show_img(img)
            if cnt_pred == 0 and gt_label != -1:  # 如果为0说明没有一个预测框和gt框匹配，即漏检
                cnt_wrong_box_miss += 1
                cnt_wrong += 1
                cnt_pred = 1
                if vis >= 1:
                    if vis == 1:
                        new_img_path = os.path.join(img_root, "wrong_box_miss", img_path[l:])
                        save_img(new_img_path, img)
                        line = new_img_path + " " + str(gt_label) + " 0\n"
                        f_wrong_box_miss.write(line)
                    elif vis == 2:
                        show_img(img)
            # if vis:
            #     plt.imshow(img[:, :, ::-1])
            #     plt.show()
            cnt_gt += cnt_pred
        # 统计误检
    false_det = 0
    det_total = 0
    print("parse the false detection")
    for idx in range(len(results)):
        if idx % 100 == 0:
            print("deal {} image".format(idx))
        gt = gts[idx]  # 一般一个idx对应一个gt框
        pred = preds[idx]
        for pred_data in pred:
            if pred_data is None:
                continue
            pred_bbox = pred_data['bbox']
            det_total += 1
            is_false_det = True
            for gt_data in gt:
                gt_box = gt_data["box"]  # x, y, w, h
                iou = IOU_XYWH(pred_bbox, gt_box)
                if iou > 0:
                    is_false_det = False
            if is_false_det:
                false_det += 1

    acc = cnt_true / float(gt_box_num)
    error = cnt_wrong / float(gt_box_num)
    error_wrong_box_miss = cnt_wrong_box_miss / float(gt_box_num)
    error_wrong_box_off = cnt_wrong_box_off / float(gt_box_num)
    error_wrong_class = cnt_wrong_class / float(gt_box_num)
    error_wrong_class_sco = cnt_wrong_class_sco / float(gt_box_num)
    # false_det = cnt_total - gt_box_num
    error_false_det = (false_det) / float(gt_box_num)

    print("cnt_gt: {}".format(cnt_gt))
    print("cnt_true: {}".format(cnt_true))
    print("cnt_wrong: {}".format(cnt_wrong))
    print("cnt_wrong_box_miss: {}".format(cnt_wrong_box_miss))
    print("cnt_wrong_box_off: {}".format(cnt_wrong_box_off))
    print("cnt_wrong_class: {}".format(cnt_wrong_class))
    print("cnt_wrong_class_sco: {}".format(cnt_wrong_class_sco))
    print("gt_box_num: {}".format(gt_box_num))
    print("false_det: {}".format(false_det))
    print("error_false_det: {}".format(error_false_det))
    print(
        "acc:{:.4f} error:{:.4f}\nbox miss:{:.4f}\nbox off:{:.4f}\nclass error {:.4f}\nclass score < {} {:.4f}\n".format(
            acc, error, error_wrong_box_miss, error_wrong_box_off, error_wrong_class, sco_thd, error_wrong_class_sco))
    print("acc| error |box miss |box off |class error |class score")
    print("{:.4f} |{:.4f} |{:.4f} |{:.4f} |{:.4f} |{:.4f}".format(acc, error, error_wrong_box_miss, error_wrong_box_off,
                                                                  error_wrong_class, error_wrong_class_sco))
    print("acc| box miss |false det")
    print("{:.4f} |{:.4f} |{:.4f}".format(acc, error_wrong_box_miss, error_false_det))
    if neg_total_num > 0:
        neg_acc = cnt_true / float(neg_total_num)
        print("neg cnt_true: {}".format(cnt_true))
        print("neg_total_num: {}".format(neg_total_num))
        print("neg_acc: {:.4f}".format(neg_acc))
    if vis:
        f_wrong.close()
        f_wrong_box_miss.close()
        f_wrong_box_off.close()
        f_wrong_class.close()
        f_wrong_class_sco.close()
        f_right.close()
    return gt_box_num


if __name__ == '__main__':
    # result_path = "/zhoudu/test/gesture/metric/yolo_for_gesture/yolov3_416_v3/epoch_270.pth/out.pkl"
    # result_path = "/zhoudu/test/gesture/metric/yolo_for_gesture/yolov3_416_v3_mobilenetv2/epoch_270.pth/out_o.pkl"
    result_path = "/zhoudu/test/gesture/metric/yolo_for_gesture/yolov3_416_v3_mobilenetv2/epoch_270.pth/out_o.pkl"
    label_paths = [
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/recog.test.ges30_imgs_1018_1019.map.box.txt",
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/neg.test.ges30_imgs_1018_1019.map.box.txt",
    ]
    # label_path = "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/neg.test.ges30_imgs_1018_1019.map.box.txt"
    save_img_roots = [
        "/zhoudu/test/gesture/show/yolov3_416_v3/imgs",
        "/zhoudu/test/gesture/show/yolov3_416_v3/imgs",
    ]
    txt_roots = [
        "/zhoudu/test/gesture/show/yolov3_416_v3/cluster_rst/",
        "/zhoudu/test/gesture/show/yolov3_416_v3/cluster_rst/"
    ]
    txt_prefixs = [
        "",
        "neg"
    ]
    vis = 1  # 0: 不可视化， 1：存储为图片 2：plt展示
    use_recog_model = True
    sco_thd = 0.0
    recog_kwarg = {}
    if use_recog_model:
        from tools.gesture_recog.build_model import build_model, get_gesture_inputs, get_test_transformer

        config = "tools/gesture_recog/exp220620c/config.py"
        model = "tools.gesture_recog.exp220620c.model.Model"
        weight_path = "/zhoudu/checkpoints/gesture_recog/gesture220620c_191544/checkpoint/model-epoch169.weights"
        recog_kwarg["config"] = config
        recog_kwarg["model"] = model
        recog_kwarg["weight_path"] = weight_path
    offset = 0
    cnt = 0
    for label_path, save_img_root, txt_root, txt_prefix in zip(label_paths, save_img_roots, txt_roots, txt_prefixs):
        print("EVAL:{}".format(label_path))
        cnt += 1
        # if cnt == 2:
        eval(result_path, label_path, vis=vis, offset=offset,
             sco_thd=sco_thd, img_root=save_img_root, txt_root=txt_root,
             txt_prefix=txt_prefix, use_recog_model=use_recog_model, **recog_kwarg)
        gt_num = len(open(label_path, "r", encoding="utf-8").readlines())
        offset += gt_num
