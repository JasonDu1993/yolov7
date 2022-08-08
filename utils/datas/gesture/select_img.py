# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 11:26
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : select_img.py
# @Software: PyCharm
import os
import json
import re
from collections import defaultdict
from utils.get_path_len import get_path_len

src_dir = "/zhoudu/golabel/gesture/label/train.ges30_cloudwalk.c5.txt"
src_img_prefix = "/zhoudu/golabel/gesture/imgs"
base_path = "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/c5/train.ges30_cloudwalk.map.box.match.txt"
base_img_prefix = "/dataset/dataset/ssd/gesture/"
dst_path = "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/sc3/train.ges30_cloudwalk.map.box.match.txt"
print("src_dir: {}".format(src_dir))
print("base_path: {}".format(base_path))
print("dst_path: {}".format(dst_path))
os.makedirs(os.path.dirname(dst_path), exist_ok=True)
select_cls = ["0", "1", "2"]
maps = defaultdict(list)
with open(base_path, "r", encoding="utf-8") as fr:
    for line in fr.readlines():
        line_sp = line.strip().split(" ")
        img_path = line_sp[0]
        img_name = img_path[get_path_len(base_img_prefix):]
        maps[img_name].append(line)
with open(dst_path, "w", encoding="utf-8") as fw:
    for name in os.listdir(src_dir):
        label = os.path.splitext(name)[0]
        if label not in select_cls:
            continue
        src_path = os.path.join(src_dir, name)
        f = open(src_path, "r", encoding="utf-8")
        paths = json.load(f)
        print("lable:{}".format(label))
        img_paths = list(paths.keys())
        print("select {} imgs".format(len(img_paths)))
        for img_path in img_paths:
            new_img_path = re.sub("box\d+", "", img_path)
            new_img_name = new_img_path[get_path_len(src_img_prefix):]
            if new_img_name in maps:
                box_index = int(re.findall("box[0-9]*", img_path)[-1][3:]) - 1 # -1是取匹配的最后一个， 3是去掉box之后只留下数字， 减1是因为需要的是index
                line = maps[new_img_name][box_index]
                # print(line)
                fw.write(line)
        f.close()
