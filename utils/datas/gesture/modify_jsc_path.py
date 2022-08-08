# -*- coding: utf-8 -*-
# @Time    : 2022/7/27 14:10
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : modify_jsc_path.py
# @Software: PyCharm
import os
import shutil

src_root = "/dataset/dataset/ssd/gesture/jiashicang/jsc220715_garage"
dst_root = "/dataset/dataset/ssd/gesture/jiashicang/jsc220715_garage_2"
for idname in sorted(os.listdir(src_root)):
    for ges in sorted(os.listdir(os.path.join(src_root, idname))):
        ges_root = os.path.join(src_root, idname, ges)
        for name in sorted(os.listdir(ges_root)):
            src_path = os.path.join(ges_root, name)
            dst_path = os.path.join(dst_root, ges, "id" + idname, ges, name)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
