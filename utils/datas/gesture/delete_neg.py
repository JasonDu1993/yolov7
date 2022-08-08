# -*- coding: utf-8 -*-
# @Time    : 2022/7/26 21:22
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : delete_neg.py
# @Software: PyCharm
src_path = "/dataset/dataset/ssd/gesture/jiashicang/c9/test.jsc220713.gt.txt"
dst_path = "/dataset/dataset/ssd/gesture/jiashicang/c9/test.jsc220713.gt.txt1"
with open(src_path, "r", encoding="utf-8") as fr:
    with open(dst_path, "w", encoding="utf-8") as fw:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            label = line_sp[1]
            if label == "-1":
                continue
            else:
                fw.write(line)
                fw.flush()