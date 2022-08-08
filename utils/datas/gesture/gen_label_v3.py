# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 11:46
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : gen_label_v2.py
# @Software: PyCharm
import os


def gen_label_v3(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r", encoding="utf-8") as fr:
        with open(dst_path, "w", encoding="utf-8") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                line_sp[1] = "0"
                new_line = " ".join(line_sp) + "\n"
                fw.write(new_line)
                fw.flush()


if __name__ == '__main__':
    src_paths = [
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v2/train.ges30_cloudwalk.ges30_imgs_1015_1017.map.box.txt",
                "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v2/test.ges30_imgs_1018_1019.map.box.txt",
    ]
    dst_paths = [
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/train.ges30_cloudwalk.ges30_imgs_1015_1017.map.box.txt",
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/test.ges30_imgs_1018_1019.map.box.txt",
        ]
    for src_path, dst_path in zip(src_paths, dst_paths):
        gen_label_v3(src_path, dst_path)
