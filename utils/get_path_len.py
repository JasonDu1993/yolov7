# -*- coding: utf-8 -*-
# @Time    : 2022/7/7 14:52
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : get_path_len.py
# @Software: PyCharm
def get_path_len(path):
    if path.endswith("/"):
        l = len(path)
    else:
        l = len(path) + 1
    return l
