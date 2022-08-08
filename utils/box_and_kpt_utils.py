# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 14:09
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : box_utils.py
# @Software: PyCharm
import os
import random
import numpy as np
import torch
import cv2

EPS = 1e-7


def square_box_to_rectangle(box):
    ratio_w = random.uniform(0.95, 1.05)
    ratio_h = random.uniform(1.1, 1.25)

    x, y, w, h = box
    new_w = w * ratio_w
    new_h = h * ratio_h
    new_x = x - (new_w - w) / 2.0
    new_y = y - (new_h - h) / 2.0
    box[0] = new_x
    box[1] = new_y
    box[2] = new_w
    box[3] = new_h
    return box


def bb_intersection_over_union(boxA, boxB):
    """

    Args:
        boxA: ndarray, x1,y1,x2,y2
        boxB:

    Returns:

    """
    # determine the (x, y)-coordinates of the intersection rectangle
    m = abs(min(np.min(boxA), np.min(boxB)))
    boxA = boxA + m
    boxB = boxB + m
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (float(boxAArea + boxBArea - interArea) + EPS)

    # return the intersection over union value
    return iou


def IOU_XYWH(Reframe, GTframe):
    """

    Args:
        Reframe: List or ndarray, len is 4, x, y, w, h
        GTframe: List or ndarray

    Returns:

    """
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]
    height1 = Reframe[3]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]
    height2 = GTframe[3]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    return ratio


def compute_total_ious(true_bbox, pred_bbox, types="onetoone"):
    assert types in ["onetoone", "onetoall"]
    true_bbox = np.array(true_bbox).reshape(-1, 4)
    pred_bbox = np.array(pred_bbox).reshape(-1, 4)
    num_true = true_bbox.shape[0]
    num_pred = pred_bbox.shape[0]
    total_ious = []
    if types.lower() == "onetoone":
        assert num_true == num_pred, "true box number {} not equal pred box number {}".format(num_true, num_pred)

        for i in range(num_true):
            ious = []
            iou = IOU_XYWH(true_bbox[i], pred_bbox[i])
            ious.append(iou)
            total_ious.append(ious)
    elif types.lower() == "onetoall":
        for i in range(num_true):
            ious = []
            for j in range(num_pred):
                iou = IOU_XYWH(true_bbox[i], pred_bbox[j])
                ious.append(iou)
            total_ious.append(ious)
    total_ious = np.array(total_ious)
    return total_ious


def get_crop_size(box, img_shape, scale=2):
    x, y, w, h = list(map(int, box))
    height, width, c = img_shape
    crop_x1 = 0
    if x - scale * w > 0:
        crop_x1 = int(x - scale * w)

    crop_x2 = width
    if x + (scale + 1) * w < width:
        crop_x2 = int(x + (scale + 1) * w)

    crop_y1 = 0
    if y - scale * h > 0:
        crop_y1 = int(y - scale * h)

    crop_y2 = height
    if y + (scale + 1) * h < height:
        crop_y2 = int(y + (scale + 1) * h)
    return crop_x1, crop_y1, crop_x2, crop_y2


def update_box(box, crop_x1, crop_y1):
    box = [box[0] - crop_x1, box[1] - crop_y1, box[2], box[3]]
    return box


def update_kpt(kpt, crop_x1, crop_y1):
    kpt = np.array(kpt)
    kpt = np.reshape(kpt, (-1, 2))  # shape [5, 2]
    kpt = kpt - np.stack([crop_x1, crop_y1])
    return kpt


def center_wh_encoding(keypoints, with_init=False, scale=(96.0, 96.0)):
    """

    Args:
      keypoints:
      with_init:
      scale: (scale_h, scale_w)

    Returns:

    """

    def _encoding(keypoints_):
        if with_init:
            tkeypoints = keypoints_ - [[0.5, 0.5], [-0.2, -0.2], [-0.2, 0.2], [0.0, 0.0], [0.2, -0.2], [0.2, 0.2]]
        else:
            tkeypoints = keypoints_ - 0.5

        tkeypoints *= torch.tensor([[scale[0], scale[1]]], dtype=torch.float32)

        return tkeypoints

    keypoints = keypoints.reshape([-1, 5, 2])

    min_y_x = torch.min(keypoints, dim=1)
    max_y_x = torch.max(keypoints, dim=1)

    min_y, min_x = torch.split(min_y_x, 2, dim=-1)
    max_y, max_x = torch.split(max_y_x, 2, dim=-1)

    center = torch.cat([(min_y + max_y) / 2.0, (min_x + max_x) / 2.0], dim=-1)

    offset = keypoints - center

    c_offset_encoding = _encoding(torch.cat([center, offset], dim=1))

    return c_offset_encoding


def get_enclose_bbox_of_kpt(kpt, kpt_num=5):
    kpt = np.array(kpt).reshape((-1, kpt_num, 2))
    x_min = np.min(kpt[:, :, 0], axis=1)
    x_max = np.max(kpt[:, :, 0], axis=1)
    y_min = np.min(kpt[:, :, 1], axis=1)
    y_max = np.max(kpt[:, :, 1], axis=1)
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    return bbox


def kpt_is_in_bbox(x, y, bbox):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    if x1 <= x <= x2 and y1 <= y <= y2:
        return True
    else:
        return False


def kpt_is_usual(x, y, bbox, scale=0.5):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    if x < x1 - scale * w or x > x2 + scale * w or y < y1 - scale * h or y > y2 + scale * h:
        return True
    else:
        return False


def get_crop_img(img, box, kpt=None, scale=2):
    crop_x1, crop_y1, crop_x2, crop_y2 = get_crop_size(box, img.shape, scale=scale)
    img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
    box = update_box(box, crop_x1, crop_y1)
    if kpt is not None:
        kpt = update_kpt(kpt, crop_x1, crop_y1)
    return img, box, kpt


def resize_box(box, in_h, in_w, out_h, out_w):
    x, y, w, h = box
    x2 = x + w
    y2 = y + h
    w_s = out_w / in_w
    h_s = out_h / in_h
    new_x = int(x * w_s)
    new_x2 = int(x2 * w_s)
    new_y = int(y * h_s)
    new_y2 = int(y2 * h_s)
    new_box = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
    return new_box


def resize_kpt(kpt, in_h, in_w, out_h, out_w):
    kpt = np.array(kpt)
    kpt = np.reshape(kpt, (-1, 2))  # shape [5, 2]
    kpt = kpt * np.stack([out_w / in_w, out_h / in_h])
    return kpt


def resize_box_and_kpt(box, kpt, in_h, in_w, out_h, out_w):
    box = resize_box(box, in_h, in_w, out_h, out_w)
    kpt = resize_kpt(kpt, in_h, in_w, out_h, out_w)
    return box, kpt


def xyxy2xywh(bbox: np.ndarray):
    """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
    evaluation.

    Args:
        bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
            ``xyxy`` order.

    Returns:
        list[float]: The converted bounding boxes, in ``xywh`` order.
    """

    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]


def get_area(box):
    x, y, w, h = box
    area = w * h
    return area


def get_box_ratio(in_h, in_w, out_h, out_w):
    ratio = min(out_h / in_h, out_w / in_w)
    out_h = int(in_h * ratio)
    out_w = int(in_w * ratio)
    return out_h,out_w, ratio

def get_resize_img(img, box, out_h, out_w, keep_ratio=True, kpt=None):
    in_h, in_w, c = img.shape
    if keep_ratio:
        out_h,out_w, ratio = get_box_ratio(in_h, in_w, out_h, out_w)

    box = np.array(box).reshape(-1, 4)
    if box.shape[0] > 1:
        new_box = []
        for b in box:
            b = resize_box(b, in_h, in_w, out_h, out_w)
            new_box.append(b)
        box = new_box
    else:
        box = [resize_box(box[0], in_h, in_w, out_h, out_w)]

    if kpt is not None:
        if isinstance(kpt, list):
            new_kpt = []
            for k in kpt:
                k = resize_kpt(k, in_h, in_w, out_h, out_w)
                new_kpt.append(k)
            kpt = new_kpt
        else:
            kpt = resize_kpt(kpt, in_h, in_w, out_h, out_w)
    img = cv2.resize(img, (out_w, out_h))
    return img, box, kpt


if __name__ == '__main__':
    # box = np.array([2.0, 3.0, 4.0, 5.0])
    # # seed = 1234
    # # random.seed(seed)
    # # square_box_to_rectangle(box)
    # # square_box_to_rectangle(box)
    #
    # kpts = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    # center_wh_encoding(kpts)
    line = "/dataset/dataset/ssd/gesture/hagrid/train_val/imgs/train_val_one/002ce59b-be3f-4002-b2fe-367f84e2e3a4.jpg 4 1200 1600 129.85 417.80 134.35 313.58"
    # "/dataset/dataset/ssd/gesture/hagrid/train_val/imgs/train_val_one/002ce59b-be3f-4002-b2fe-367f84e2e3a4.jpg 0 1200 1600 144.68 415.95 120.49 248.92"
    line_sp = line.strip().split(" ")
    img_path = line_sp[0]
    img = cv2.imread(img_path)
    base_box = list(map(float, line_sp[4:4 + 4]))
    get_resize_img(img, base_box, 416, 416)
