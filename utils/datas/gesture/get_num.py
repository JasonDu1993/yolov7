# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 15:55
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : get_num.py
# @Software: PyCharm
from collections import defaultdict

id_to_path = defaultdict(list)
path = "/dataset/dataset/ssd/gesture/jiashicang/c9/test.jsc220713.map.box.match.txt"
with open(path, "r", encoding="utf-8") as fr:
    for line in fr.readlines():
        line_sp = line.strip().split(" ")
        img_path = line_sp[0]
        img_id = line_sp[1]
        id_to_path[img_id].append(img_path)

for i in sorted(id_to_path.keys()):
    print(len(id_to_path[i]))