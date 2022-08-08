# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 18:34
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : rotate_or_trunc.py
# @Software: PyCharm
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from gesture_recog.transformer.rotate import RotateCombination
from gesture_recog.transformer.trunc import TruncTransformer

kpt_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
             (0, 255, 255)]  # left_eye(green), right_eye(red), nose(blue), left_mouth(cyan), right_mouth(yellow)


def normalize_kpt_with_bbox(kpt, bbox):
    kpt = np.array(kpt).reshape((-1, 2))
    x, y, w, h = bbox
    kpt_norm = (kpt - np.array([x, y])) / np.stack([w, h])
    return kpt_norm


class RotateOrTrunc(object):
    def __init__(self, p=0.3, min_angle=-30, max_angle=30, class_weight=None,
                 start_ratio=0.0, end_ratio=0.4, num=2, p2=0.2, must_cut_p=0.5,
                 seed=1234, debug=False):
        self.p = p
        self.rng = np.random.RandomState(seed)
        self.debug = debug
        self.rotate = RotateCombination(p=1, min_angle=min_angle, max_angle=max_angle, class_weight=class_weight,
                                        seed=seed, debug=debug)
        self.trunc = TruncTransformer(p=1, start_ratio=start_ratio, end_ratio=end_ratio, num=num, p2=p2,
                                      must_cut_p=must_cut_p, seed=seed, debug=debug)

    def __call__(self, img, bbox, kpt, path=None):
        p = self.rng.random()
        if p < self.p:
            p2 = self.rng.random()
            if p2 < 0.5:
                kpts_norm = normalize_kpt_with_bbox(kpt, bbox)
                eye_v_dis = abs(kpts_norm[1][1] - kpts_norm[0][1])
                if eye_v_dis < 0.3:
                    img, bbox, kpt = self.rotate(img, bbox, kpt)
            else:
                img, bbox, kpt = self.trunc(img, bbox, kpt)
        return img, bbox, kpt, path
