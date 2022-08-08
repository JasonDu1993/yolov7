# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 15:48
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : t.py
# @Software: PyCharm
src_path = "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v1/train.ges30_imgs_1018_1019.map.box.txt"
dst_path = src_path + "1"
with open(src_path, "r", encoding="utf-8") as fr:
    with open(dst_path, "w", encoding="utf-8") as fw:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            # line_sp[1] = "-1"
            img_w = line_sp[-2]
            img_h = line_sp[-1]
            line_sp.pop()
            line_sp.pop()
            line_sp.insert(2, img_w)
            line_sp.insert(3, img_h)
            new_line = " ".join(line_sp) + "\n"
            print(new_line)
            fw.write(new_line)
            fw.flush()
import shutil

shutil.move(dst_path, src_path)
