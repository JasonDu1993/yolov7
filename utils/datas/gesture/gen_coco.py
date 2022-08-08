# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 17:40
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : gen_coco.py
# @Software: PyCharm
import os.path as osp
import cv2
import json
from time import time
from utils.get_path_len import get_path_len


def get_cat(class_map_path):
    labels = []
    with open(class_map_path, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            id = int(line_sp[0])
            name = line_sp[1]
            data = {"id": id, "name": name, "supercategory": "gesture"}
            labels.append(data)
    return labels


def convert_to_coco(src_path, dst_path, class_map_path, image_prefix=None, modify_label=False):
    annotations = []
    images = []
    obj_count = 0
    if image_prefix is None:
        l = 0
    else:
        l = get_path_len(image_prefix)
    cats = get_cat(class_map_path)
    with open(src_path, "r", encoding="utf-8") as fr:
        for idx, line in enumerate(fr.readlines()):
            line_sp = line.strip().split(" ")
            img_path = line_sp[0][l:]
            label = line_sp[1]
            if modify_label:
                label = 0  # 由于现在不区分手的类别因此只有一个类，标签为0
            box = line_sp[4:8]  # x y w h
            x, y, w, h = list(map(float, box))
            # img = cv2.imread(img_path)
            #  = img.shape[:2]
            width, height = list(map(int, line_sp[2:4]))  # 图片的宽高

            images.append(dict(
                id=idx,
                file_name=img_path,
                height=height,
                width=width))

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=int(label),
                bbox=[x, y, w, h],
                area=w * h,
                segmentation=[],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=cats)
    with open(dst_path, "w", encoding="utf-8") as fw:
        json.dump(coco_format_json, fw)


if __name__ == '__main__':

    src_paths = [
        "/data/gesture/jiashicang/resize/txtv2/train.jsc220713.noneg.gt.txt",
    ]
    dst_paths = [
        "/data/gesture/jiashicang/resize/txtv2/train.jsc220713.noneg.gt.json",
    ]
    class_map_path = "mmdet/datasets/gesture/desc/class_description_v3.txt"
    image_prefix = None
    modify_label = False
    for src_path, dst_path in zip(src_paths, dst_paths):
        convert_to_coco(src_path, dst_path, class_map_path, image_prefix, modify_label=modify_label)
