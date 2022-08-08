# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 20:15
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : modify_box.py
# @Software: PyCharm
import cv2
import numpy as np
from utils.box_and_kpt_utils import get_box_ratio, resize_box, get_resize_img

src_path = "/dataset/dataset/ssd/gesture/jiashicang/resize/txtv2/test.jsc220713.gt.txt"
dst_path = "/dataset/dataset/ssd/gesture/jiashicang/resize/txtv2/test.jsc220713.gt.txt1"
out_shape = [416, 416]  # out_h, out_w
with open(src_path, "r", encoding="utf-8") as fr:
    with open(dst_path, "w", encoding="utf-8") as fw:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            label = line_sp[1]
            in_w = int(line_sp[2])
            in_h = int(line_sp[3])
            out_h = out_shape[0]
            out_w = out_shape[1]
            out_h, out_w, ratio = get_box_ratio(in_h, in_w, out_h, out_w)
            bbox = list(map(float, line_sp[4:8]))
            new_box = resize_box(bbox, in_h, in_w, out_h, out_w)
            bbox_str = " ".join(list(map(lambda x: "{:.2f}".format(x), new_box)))
            new_line = img_path + " " + str(label) + " " + str(out_w) + " " + str(
                out_h) + " " + bbox_str + "\n"
            fw.write(new_line)
