# -*- coding: utf-8 -*-
# @Time    : 2022/6/24 16:34
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : gen_neg.py
# @Software: PyCharm
import os


def gen_neg_txt(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r", encoding="utf-8") as fr:
        with open(dst_path, "w", encoding="utf-8") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                label = line_sp[1]
                if label == "0" and "负例" in img_path:  # mmdet/datasets/gesture/desc/class_description.txt, 0 is neg
                    line_sp[1] = "-1"
                    new_line = " ".join(line_sp) + "\n"
                    fw.write(new_line)
                    fw.flush()
if __name__ == '__main__':
    src_paths = [
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v1/train.ges30_cloudwalk.map.box.txt",
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v1/train.ges30_imgs_1015_1017.map.box.txt",
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v1/train.ges30_imgs_1018_1019.map.box.txt",
    ]
    dst_paths = [
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/neg.train.ges30_cloudwalk.map.box.txt",
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/neg.train.ges30_imgs_1015_1017.map.box.txt",
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/neg.train.ges30_imgs_1018_1019.map.box.txt",
    ]

    for src_path, dst_path in zip(src_paths, dst_paths):
        gen_neg_txt(src_path, dst_path)