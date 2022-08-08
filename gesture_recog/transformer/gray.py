# -*- coding: utf-8 -*-
# @Time    : 2022/1/4 16:52
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : black.py
# @Software: PyCharm
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.draw_box_kpt_utils import draw_box_kpt_with_cwface_label_over_image
from gesture_recog.transformer.color import color_aug

kpt_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
             (0, 255, 255)]  # left_eye(green), right_eye(red), nose(blue), left_mouth(cyan), right_mouth(yellow)


class GrayTransformer(object):
    def __init__(self, p=0.5, scope=0.125, seed=1234, mode=2, debug=False):
        self.rng = np.random.RandomState(seed)
        self.debug = debug
        self.p = p
        self.scope = scope
        self.mode = mode
        self.red_channel = 1

    def __call__(self, img, bbox=None, kpt=None, path=None):
        """kpt的坐标是相对于img的，bbox则是原图上的坐标

        Args:
            img:
            bbox:
            kpt:

        Returns:

        """
        p = self.rng.random()
        if p < self.p:
            # img1 = copy.deepcopy(img)
            # img_draw1 = draw_box_kpt_with_cwface_label_over_image(img1, [bbox], kpt)
            # plt.imshow(img_draw1[:, :, ::-1])
            # plt.show()
            img = np.copy(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # im_red = img[:, :, self.red_channel]
            img = np.repeat(img[:, :, None], repeats=3, axis=-1)
            p2 = self.rng.random()
            if p2 > 0.5:
                img = img.astype(np.float32)
                img = color_aug(img, self.scope, self.rng, self.mode)
            if self.debug:
                img = img.astype(np.uint8)
                img_draw_2 = draw_box_kpt_with_cwface_label_over_image(img, [bbox], kpt)
                plt.imshow(img_draw_2[:, :, ::-1])
                plt.show()
        return img, bbox, kpt, path


if __name__ == '__main__':
    img = cv2.imread("../test/3.jpg")
    t = GrayTransformer(p=1, scope=0.2, debug=True)
    bbox = [15, 22, 88, 95]
    kpt = [44, 65, 87, 63, 67, 89, 52, 102, 81, 100]
    kpt = np.array(kpt).reshape(5, 2)
    # for i, pt in enumerate(kpt):
    #     cv2.circle(img, (int(pt[0]), int(pt[1])), 3, kpt_color[i], -1)
    # plt.imshow(img[:, :, ::-1])
    # plt.show()
    m = t(img, bbox, kpt)
