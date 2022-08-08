# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 15:24
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : get_heart_img.py
# @Software: PyCharm
import os

cls_name = "c9"
src_paths = [
    "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/train.ges30_cloudwalk.map.box.match.txt",
    "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/train.ges30_imgs_1015_1017.map.box.match.txt",
    "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/train.ges30_imgs_1018_1019.map.box.match.txt",
    "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/" + cls_name + "/train.indoor_multi_scenario.map.box.match.txt",
    "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/" + cls_name + "/train.indoor_office.map.box.match.txt",
]
dst_path = "/dataset/dataset/ssd/gesture/hagrid/c9/heart.txt"
dst_path_rep = "/dataset/dataset/ssd/gesture/hagrid/c9/heart.rep.txt"
os.makedirs(os.path.dirname(dst_path), exist_ok=True)
with open(dst_path, "w", encoding="utf-8") as fw:
    for src_path in src_paths:
        with open(src_path, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                label = line_sp[1]
                if label == "7":
                    fw.write(line)
                    fw.flush()

with open(dst_path_rep, "w", encoding="utf-8") as fw:
    for i in range(9):
        with open(dst_path, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                fw.write(line)
                fw.flush()
