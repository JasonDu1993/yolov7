# -*- coding: utf-8 -*-
# @Time    : 2021/10/21 14:41
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : crop_and_padding.py
# @Software: PyCharm
import cv2
import numpy as np


class ExpandBbox(object):
    def __init__(self, inacc_box=0.0, ratio=None, seed=1234):
        self.rng = np.random.RandomState(seed)
        self.ratio = ratio
        self.ratio_w = 1
        self.ratio_h = 1.2
        self.inacc_box = inacc_box
        print("inacc_box:{}".format(inacc_box))
        if self.ratio is not None:
            self.ratio_w, self.ratio_h = ratio
        else:
            print("The self.ratio_w, self.ratio_h will random")

    def __call__(self, img, bbox, kpt=None, path=None):
        """kpt的坐标是相对于img的，bbox则是原图上的坐标

        Args:
            img:
            bbox:
            kpt:

        Returns:

        """
        if self.ratio is None:
            self.ratio_w = float(self.rng.uniform(0.9, 1.1))
            self.ratio_h = float(self.rng.uniform(0.9, 1.1))
        # print(self.ratio_w, self.ratio_h)
        # print(bbox)
        x, y, w, h = bbox
        if self.inacc_box:
            p = self.rng.uniform(0, 1)
        else:
            p = 0
        if p <= 1 - self.inacc_box:
            new_w = int(w * self.ratio_w)
            new_h = int(h * self.ratio_h)
            new_x = int(x - (new_w - w) * 0.5)
            new_y = int(y - (new_h - h) * 0.5)
            bbox = [new_x, new_y, new_w, new_h]
        else:
            new_w = int(w * self.ratio_w)
            new_h = int(h * self.ratio_h)
            s_w = self.rng.uniform(0.45, 0.55)
            s_h = self.rng.uniform(0.45, 0.55)
            new_x = int(x - (new_w - w) * s_w)
            new_y = int(y - (new_h - h) * s_h)
            bbox = [new_x, new_y, new_w, new_h]
        # print(bbox)
        return img, bbox, kpt, path


if __name__ == '__main__':
    img = cv2.imread("3.jpg")
    t = ExpandBbox()
    bbox = [10, -5, 100, 150]
    kpt = [[10, 2, 100, 2, 50, 50, 110, 10, 120, 90]]
    m = t(img, bbox, kpt)
    m = t(img, bbox, kpt)
