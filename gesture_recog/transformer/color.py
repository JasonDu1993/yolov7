# -*- coding: utf-8 -*-
# @Time    : 2021/10/21 14:41
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : crop_and_padding.py
# @Software: PyCharm
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.draw_box_kpt_utils import draw_box_kpt_with_cwface_label_over_image

kpt_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
             (0, 255, 255)]  # left_eye(green), right_eye(red), nose(blue), left_mouth(cyan), right_mouth(yellow)


def brightness_aug(src, x, rng):
    alpha = 1.0 + float(rng.uniform(-x, x))
    src *= alpha
    return src


def contrast_aug(src, x, rng):
    alpha = 1.0 + float(rng.uniform(-x, x))
    coef = np.array([[[0.299, 0.587, 0.114]]])
    gray = src * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    src *= alpha
    src += gray
    return src


def saturation_aug(src, x, rng):
    alpha = 1.0 + float(rng.uniform(-x, x))
    coef = np.array([[[0.299, 0.587, 0.114]]])
    gray = src * coef
    gray = np.sum(gray, axis=2, keepdims=True)
    gray *= (1.0 - alpha)
    src *= alpha
    src += gray
    return src


def color_aug(img, x, rng, mode=2):
    if mode > 1:
        augs = [brightness_aug, contrast_aug, saturation_aug]
        rng.shuffle(augs)
    else:
        augs = [brightness_aug]
        # augs = [contrast_aug]
    for aug in augs:
        # print(img.shape)
        img = aug(img, x, rng)
        # print(img.shape)
    return img


class ColorTransformer(object):
    def __init__(self, p=0.5, scope=0.125, seed=1234, mode=2, debug=False):
        self.rng = np.random.RandomState(seed)
        self.debug = debug
        self.p = p
        self.scope = scope
        self.mode = mode

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
            img = img.astype(np.float32)
            img = color_aug(img, self.scope, self.rng, self.mode)
            # img = img.astype(np.uint8)
            if self.debug:
                img_draw_2 = draw_box_kpt_with_cwface_label_over_image(img, [bbox], kpt)
                plt.imshow(img_draw_2[:, :, ::-1])
                plt.show()
        return img, bbox, kpt, path


if __name__ == '__main__':
    img = cv2.imread("3.jpg")
    t = ColorTransformer(p=1, debug=True)
    bbox = [15, 22, 88, 95]
    kpt = [44, 65, 87, 63, 67, 89, 52, 102, 81, 100]
    kpt = np.array(kpt).reshape(5, 2)
    # for i, pt in enumerate(kpt):
    #     cv2.circle(img, (int(pt[0]), int(pt[1])), 3, kpt_color[i], -1)
    # plt.imshow(img[:, :, ::-1])
    # plt.show()
    m = t(img, bbox, kpt)
