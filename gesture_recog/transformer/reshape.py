# -*- coding: utf-8 -*-
# @Time    : 2021/10/21 15:27
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : reshape.py
# @Software: PyCharm
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.draw_box_kpt_utils import draw_box_kpt_with_cwface_label_over_image


class ReshapeTransformer(object):
    def __init__(self, input_shape, kpt_type="regression", debug=False):
        self.c, self.h, self.w = input_shape
        assert self.c in [1, 3]
        self.kpt_type = kpt_type
        self.debug = debug
        if self.debug:
            from datetime import datetime

            TIMESTAMP = datetime.now().strftime("%y%m%d%H%M%S")
            self.log_path = os.path.join("/dataset/dataset/ssd/kpt_syn/cluster_rst", "reshape_log_" + TIMESTAMP + ".txt")

    def __call__(self, img, bbox, kpt=None, path=None):
        """kpt的坐标是相对于img的，bbox则是原图上的坐标

        Args:
            img:
            bbox:
            kpt:

        Returns:

        """
        h, w, c = img.shape
        assert c == 3, c
        if self.c == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if h != self.h or w != self.w:
            img = cv2.resize(img, (self.w, self.h))
            if kpt is not None:
                bbox_tmp = np.array(bbox)
                kpt_tmp = np.array(kpt)
                if self.kpt_type == "regression":
                    kpt = np.array(kpt)
                    kpt = np.reshape(kpt, (-1, 2))  # shape [5, 2]
                    kpt = kpt * np.stack([self.w / w, self.h / h])
                elif self.kpt_type == 'heatmap':
                    kpt = kpt
        if self.debug:
            img_draw = draw_box_kpt_with_cwface_label_over_image(img, [bbox], kpt, draw_box=False)
            plt.imshow(img_draw[:, :, ::-1])
            plt.show()
        return img, bbox, kpt, path


if __name__ == '__main__':
    img = cv2.imread("3.jpg")
    t = ReshapeTransformer((3, 64, 64))
    kpt = [[10, 2, 100, 2, 50, 50, 110, 10, 120, 90]]
    img = t(img, kpt)
