# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 17:30
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : merge.py
# @Software: PyCharm
import os
from collections import defaultdict
from natsort import natsorted


def get_merge_txt_sort(paths, dst_path):
    info = defaultdict(list)
    for path in paths:
        with open(path, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                label = line_sp[1]
                info[label].append(line)

    with open(dst_path, "w", encoding="utf-8") as fw:
        for key in natsorted(info):
            value = info[key]
            for line in natsorted(value):
                fw.write(line)


def get_merge_txt_seq(paths, dst_path, modify_label_id=False):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as fw:
        for path in paths:
            with open(path, "r", encoding="utf-8") as fr:
                for line in fr.readlines():
                    if modify_label_id:
                        line_sp = line.strip().split(" ")
                        line_sp[1] = "0"
                        new_line = " ".join(line_sp) + "\n"
                        fw.write(new_line)
                    else:
                        fw.write(line)


if __name__ == '__main__':
    # paths = [
    #     "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/recog.test.ges30_imgs_1018_1019.map.box.txt",
    #     "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/neg.test.ges30_imgs_1018_1019.map.box.txt",
    # ]
    # dst_path = "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/merge.ges30_imgs_1018_1019.txt"
    # get_merge_txt_sort(paths, dst_path)

    paths = [
        "/data/gesture/common/resize/txt/train.ges30_cloudwalk.ges30_imgs_1015_1017.map.box.txt",
        "/data/gesture/hagrid/resize/txt/train.hagrid.gt.txt",
        "/data/gesture/jiashicang/resize/txtv2/train.jsc220713.gt.txt",
    ]
    dst_path = "/data/gesture/merge/merge.ges30_cloudwalk.ges30_imgs_1015_1017.hagrid.jsc220713.box.txt"
    modify_label_id = True
    get_merge_txt_seq(paths, dst_path, modify_label_id=modify_label_id)
