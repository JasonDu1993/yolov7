# -*- coding: utf-8 -*-
# @Time    : 2021/11/11 14:42
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : trunc.py
# @Software: PyCharm
import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import bisect

kpt_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
             (0, 255, 255)]  # left_eye(green), right_eye(red), nose(blue), left_mouth(cyan), right_mouth(yellow)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.draw_box_kpt_utils import draw_box_kpt_with_cwface_label_over_image

from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])


class CutTopFace(object):
    def __init__(self, start_ratio=0.0, end_ratio=0.4, must_cut=False, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug
        self.must_cut = must_cut

    def __call__(self, img, bbox, kpt=None, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        ratio = self.rng.uniform(self.start_ratio, self.end_ratio)
        cut_h = int(h * ratio)
        if self.must_cut:
            kpt = np.array(kpt).reshape(-1, 2)
            kpt_y1 = np.min(kpt[:, 1])
            kpt_y2 = np.max(kpt[:, 1])
            r = self.rng.uniform(self.start_ratio, self.end_ratio)
            if kpt_y1 < y or kpt_y2 > y + h:
                r1 = self.rng.uniform(0, 0.15)
                cut_h = int(r1 * h)
            else:
                cut_h = int(kpt_y1 - min(y, kpt_y1) + (kpt_y2 - kpt_y1) * r)
        new_h = h - cut_h
        new_y = y + cut_h
        img_new = img[new_y:]
        bbox = [x, 0, w, new_h]
        if kpt is not None:
            kpt = np.array(kpt)
            kpt = np.reshape(kpt, (-1, 2))  # shape [5, 2]
            kpt = kpt - np.stack([0, new_y])
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class CutBottomFace(object):
    def __init__(self, start_ratio=0.0, end_ratio=0.4, must_cut=False, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug
        self.must_cut = must_cut

    def __call__(self, img, bbox, kpt=None, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        ratio = self.rng.uniform(self.start_ratio, self.end_ratio)
        cut_h = int(h * ratio)
        if self.must_cut:
            kpt = np.array(kpt).reshape(-1, 2)
            kpt_y1 = np.min(kpt[:, 1])
            kpt_y2 = np.max(kpt[:, 1])
            r = self.rng.uniform(self.start_ratio, self.end_ratio)
            if kpt_y1 < y or kpt_y2 > y + h:
                r1 = self.rng.uniform(0, 0.15)
                cut_h = int(r1 * h)
            else:
                cut_h = int(max(y + h, kpt_y2) - kpt_y2 + (kpt_y2 - kpt_y1) * r)
        new_h = h - cut_h
        new_y2 = y + new_h
        img_new = img[:new_y2]
        bbox = [x, y, w, new_h]
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class CutLeftFace(object):
    def __init__(self, start_ratio=0.0, end_ratio=0.4, must_cut=False, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug
        self.must_cut = must_cut

    def __call__(self, img, bbox, kpt=None, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        ratio = self.rng.uniform(self.start_ratio, self.end_ratio)
        cut_w = int(w * ratio)
        if self.must_cut:
            kpt = np.array(kpt).reshape(-1, 2)
            kpt_x1 = np.min(kpt[:, 0])
            kpt_x2 = np.max(kpt[:, 0])
            r = self.rng.uniform(self.start_ratio, self.end_ratio)
            if kpt_x1 < x or kpt_x2 > x + w:
                r1 = self.rng.uniform(0, 0.15)
                cut_w = int(r1 * w)
            else:
                cut_w = int(kpt_x1 - min(x, kpt_x1) + (kpt_x2 - kpt_x1) * r)
        new_w = w - cut_w
        new_x = x + cut_w
        img_new = img[:, new_x:]
        bbox = [0, y, new_w, h]
        if kpt is not None:
            kpt = np.array(kpt)
            kpt = np.reshape(kpt, (-1, 2))  # shape [5, 2]
            kpt = kpt - np.stack([new_x, 0])
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class CutRightFace(object):
    def __init__(self, start_ratio=0.0, end_ratio=0.4, must_cut=False, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug
        self.must_cut = must_cut

    def __call__(self, img, bbox, kpt=None, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        ratio = self.rng.uniform(self.start_ratio, self.end_ratio)
        cut_w = int(w * ratio)
        if self.must_cut:
            kpt = np.array(kpt).reshape(-1, 2)
            kpt_x1 = np.min(kpt[:, 0])
            kpt_x2 = np.max(kpt[:, 0])
            r = self.rng.uniform(self.start_ratio, self.end_ratio)
            if kpt_x1 < x or kpt_x2 > x + w:
                r1 = self.rng.uniform(0, 0.15)
                cut_w = int(r1 * w)
            else:
                cut_w = int(max(x + w, kpt_x2) - kpt_x2 + (kpt_x2 - kpt_x1) * r)
        new_w = w - cut_w
        new_x2 = x + new_w
        img_new = img[:, :new_x2]
        bbox = [x, y, new_w, h]
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class RetainLeftEye(object):
    def __init__(self, start_ratio=0.0, end_ratio=1.0, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug

    def __call__(self, img, bbox, kpt, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        img_h, img_w, c = img.shape
        x2 = x + w
        y2 = y + h
        kpt = np.array(kpt).reshape(-1, 2)
        left_eye = Point(kpt[0][0], kpt[0][1])
        nose = Point(kpt[2][0], kpt[2][1])
        r_x = self.rng.uniform(self.start_ratio, self.end_ratio)
        r_y = self.rng.uniform(self.start_ratio, self.end_ratio)
        l_x = (nose.x - left_eye.x) * r_x
        l_y = (nose.y - left_eye.y) * r_y
        if left_eye.x < nose.x:  # 鼻子在左眼右边
            new_x2 = min(img_w, int(left_eye.x + l_x))
            new_y2 = min(img_h, int(left_eye.y + l_y))
            img_new = img[:new_y2, :new_x2]
            bbox = [x, y, new_x2 - x, new_y2 - y]
        else:  # 鼻子在左眼左边
            new_x = max(0, int(left_eye.x + l_x))
            new_y2 = min(img_h, int(left_eye.y + l_y))
            img_new = img[:new_y2, new_x:]
            bbox = [0, y, x2 - new_x, new_y2 - y]

            kpt = kpt - np.stack([new_x, 0])
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class RetainRightEye(object):
    def __init__(self, start_ratio=0.0, end_ratio=1.0, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug

    def __call__(self, img, bbox, kpt, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        img_h, img_w, c = img.shape
        x2 = x + w
        y2 = y + h
        kpt = np.array(kpt).reshape(-1, 2)
        right_eye = Point(kpt[1][0], kpt[1][1])
        nose = Point(kpt[2][0], kpt[2][1])
        r_x = self.rng.uniform(self.start_ratio, self.end_ratio)
        r_y = self.rng.uniform(self.start_ratio, self.end_ratio)

        l_x = (right_eye.x - nose.x) * r_x
        l_y = (nose.y - right_eye.y) * r_y
        if right_eye.x > nose.x:  # 鼻子在右眼左边
            new_x = max(0, int(right_eye.x - l_x))
            new_y2 = min(img_h, int(right_eye.y + l_y))
            img_new = img[:new_y2, new_x:]
            bbox = [0, y, x2 - new_x, new_y2 - y]

            kpt = kpt - np.stack([new_x, 0])
        else:  # 鼻子在右眼左边
            new_x2 = min(img_w, int(right_eye.x - l_x))
            new_y2 = min(img_h, int(right_eye.y + l_y))
            img_new = img[:new_y2, :new_x2]
            bbox = [x, y, new_x2 - x, new_y2 - y]
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class RetainLeftMouth(object):
    def __init__(self, start_ratio=0.0, end_ratio=1.0, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug

    def __call__(self, img, bbox, kpt, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        img_h, img_w, c = img.shape
        x2 = x + w
        y2 = y + h
        kpt = np.array(kpt).reshape(-1, 2)
        nose = Point(kpt[2][0], kpt[2][1])
        left_mouth = Point(kpt[3][0], kpt[3][1])
        r_x = self.rng.uniform(self.start_ratio, self.end_ratio)
        r_y = self.rng.uniform(self.start_ratio, self.end_ratio)

        l_x = (nose.x - left_mouth.x) * r_x
        l_y = (left_mouth.y - nose.y) * r_y
        if left_mouth.x < nose.x:  # 鼻子在左嘴角右边
            new_x2 = min(img_w, int(left_mouth.x + l_x))
            new_y = max(0, int(left_mouth.y - l_y))
            img_new = img[new_y:, :new_x2]
            bbox = [x, 0, new_x2 - x, y2 - new_y]

            kpt = kpt - np.stack([0, new_y])
        else:  # 鼻子在左嘴角左边
            new_x = max(0, int(left_mouth.x + l_x))
            new_y = max(0, int(left_mouth.y - l_y))
            img_new = img[new_y:, new_x:]
            bbox = [0, 0, x2 - new_x, y2 - new_y]

            kpt = kpt - np.stack([new_x, new_y])
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class RetainRightMouth(object):
    def __init__(self, start_ratio=0.0, end_ratio=1.0, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug

    def __call__(self, img, bbox, kpt, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        img_h, img_w, c = img.shape
        x2 = x + w
        y2 = y + h
        kpt = np.array(kpt).reshape(-1, 2)
        nose = Point(kpt[2][0], kpt[2][1])
        right_mouth = Point(kpt[4][0], kpt[4][1])
        r_x = self.rng.uniform(self.start_ratio, self.end_ratio)
        r_y = self.rng.uniform(self.start_ratio, self.end_ratio)

        l_x = (right_mouth.x - nose.x) * r_x
        l_y = (right_mouth.y - nose.y) * r_y
        if right_mouth.x > nose.x:  # 鼻子在右嘴角左边
            new_x = max(0, int(right_mouth.x - l_x))
            new_y = max(0, int(right_mouth.y - l_y))
            img_new = img[new_y:, new_x:]
            bbox = [0, 0, x2 - new_x, y2 - new_y]

            kpt = kpt - np.stack([new_x, new_y])
        else:  # 鼻子在右嘴角右边
            new_x2 = min(img_w, int(right_mouth.x - l_x))
            new_y = max(0, int(right_mouth.y - l_y))
            img_new = img[new_y:, :new_x2]
            bbox = [x, 0, new_x2 - x, y2 - new_y]

            kpt = kpt - np.stack([0, new_y])

        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class RetainLeftEyeAndLeftMouth(object):
    def __init__(self, start_ratio=0.0, end_ratio=1.0, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug

    def __call__(self, img, bbox, kpt, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        img_h, img_w, c = img.shape
        x2 = x + w
        y2 = y + h
        kpt = np.array(kpt).reshape(-1, 2)
        left_eye = Point(kpt[0][0], kpt[0][1])
        nose = Point(kpt[2][0], kpt[2][1])
        left_mouth = Point(kpt[3][0], kpt[3][1])
        max_x = max(left_eye.x, left_mouth.x)
        min_x = min(left_eye.x, left_mouth.x)
        r_x = self.rng.uniform(self.start_ratio, self.end_ratio)
        if nose.x > min_x:
            l_x = (x2 - max_x) * r_x
            new_x2 = min(img_w, int(max_x + l_x))
            img_new = img[:, :new_x2]
            bbox = [x, y, new_x2 - x, y2 - y]
        else:
            l_x = (min_x - x) * r_x
            new_x = max(0, int(min_x - l_x))

            img_new = img[:, new_x:]
            bbox = [0, y, x2 - new_x, y2 - y]

            kpt = kpt - np.stack([new_x, 0])
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class RetainRightEyeAndRightMouth(object):
    def __init__(self, start_ratio=0.0, end_ratio=0.4, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug

    def __call__(self, img, bbox, kpt, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        img_h, img_w, c = img.shape
        x2 = x + w
        y2 = y + h
        kpt = np.array(kpt).reshape(-1, 2)
        right_eye = Point(kpt[1][0], kpt[1][1])
        nose = Point(kpt[2][0], kpt[2][1])
        right_mouth = Point(kpt[4][0], kpt[4][1])
        min_x = min(right_eye.x, right_mouth.x)
        max_x = min(right_eye.x, right_mouth.x)
        r_x = self.rng.uniform(self.start_ratio, self.end_ratio)
        r_y = self.rng.uniform(self.start_ratio, self.end_ratio)

        if nose.x < min_x:
            l_x = (min_x - x) * r_x
            new_x = max(0, int(min_x - l_x))
            img_new = img[:, new_x:]
            bbox = [0, y, x2 - new_x, y2 - y]

            kpt = kpt - np.stack([new_x, 0])
        else:
            l_x = (x2 - max_x) * r_x
            new_x2 = min(img_w, int(max_x + l_x))
            img_new = img[:, :new_x2]
            bbox = [x, y, new_x2 - x, y2 - y]
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class RetainLeftEyeAndRightEye(object):
    def __init__(self, start_ratio=0.1, end_ratio=0.4, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug

    def __call__(self, img, bbox, kpt, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        img_h, img_w, c = img.shape
        x2 = x + w
        y2 = y + h
        kpt = np.array(kpt).reshape(-1, 2)
        left_eye = Point(kpt[0][0], kpt[0][1])
        right_eye = Point(kpt[1][0], kpt[1][1])
        max_y = max(left_eye.y, right_eye.y)
        r_x = self.rng.uniform(self.start_ratio, self.end_ratio)
        r_y = self.rng.uniform(self.start_ratio, self.end_ratio)
        new_y2 = min(img_h, int(max_y + (y2 - max_y) * r_y))
        img_new = img[:new_y2, :]
        bbox = [x, y, x2 - x, new_y2 - y]
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class RetainLeftMouthAndRightMouth(object):
    def __init__(self, start_ratio=0.0, end_ratio=1.0, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug

    def __call__(self, img, bbox, kpt, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        img_h, img_w, c = img.shape
        x2 = x + w
        y2 = y + h
        kpt = np.array(kpt).reshape(-1, 2)
        nose = Point(kpt[2][0], kpt[2][1])
        left_mouth = Point(kpt[3][0], kpt[3][1])
        right_mouth = Point(kpt[4][0], kpt[4][1])
        min_y = min(left_mouth.y, right_mouth.y)
        r_y = self.rng.uniform(self.start_ratio, self.end_ratio)
        r_y_ = self.rng.uniform(0, 0.4)
        l = min_y - nose.y if min_y > nose.y else (y2 - y) * r_y_
        new_y = max(0, int(min_y - l * r_y))

        img_new = img[new_y:, :]
        bbox = [x, 0, x2 - x, y2 - new_y]

        kpt = kpt - np.stack([0, new_y])
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class RetainLeftEyeAndNose(object):
    def __init__(self, start_ratio=0.0, end_ratio=1.0, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug

    def __call__(self, img, bbox, kpt, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        img_h, img_w, c = img.shape
        x2 = x + w
        y2 = y + h
        kpt = np.array(kpt).reshape(-1, 2)
        left_eye = Point(kpt[0][0], kpt[0][1])
        nose = Point(kpt[2][0], kpt[2][1])
        r_x = self.rng.uniform(self.start_ratio, self.end_ratio)
        r_y = self.rng.uniform(self.start_ratio, self.end_ratio)
        if left_eye.x < nose.x:  # 鼻子在左眼右边
            l_x = (x2 - nose.x) * r_x
            l_y = (y2 - nose.y) * r_y
            new_x2 = min(img_w, int(nose.x + l_x))
            new_y2 = min(img_h, int(nose.y + l_y))
            img_new = img[:new_y2, :new_x2]
            bbox = [x, y, new_x2 - x, new_y2 - y]
        else:  # 鼻子在左眼左边
            l_x = (nose.x - x) * r_x
            l_y = (y2 - nose.y) * r_y
            new_x = max(0, int(nose.x - l_x))
            new_y2 = min(img_h, int(nose.y + l_y))
            img_new = img[:new_y2, new_x:]
            bbox = [0, y, x2 - new_x, new_y2 - y]

            kpt = kpt - np.stack([new_x, 0])
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class RetainRightEyeAndNose(object):
    def __init__(self, start_ratio=0.0, end_ratio=1.0, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug

    def __call__(self, img, bbox, kpt, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        img_h, img_w, c = img.shape
        x2 = x + w
        y2 = y + h
        kpt = np.array(kpt).reshape(-1, 2)
        right_eye = Point(kpt[1][0], kpt[1][1])
        nose = Point(kpt[2][0], kpt[2][1])
        r_x = self.rng.uniform(self.start_ratio, self.end_ratio)
        r_y = self.rng.uniform(self.start_ratio, self.end_ratio)
        if right_eye.x > nose.x:  # 鼻子在右眼左边
            l_x = (nose.x - x) * r_x
            l_y = (y2 - nose.y) * r_y
            new_x = max(0, int(nose.x - l_x))
            new_y2 = min(img_h, int(nose.y + l_y))
            img_new = img[:new_y2, new_x:]
            bbox = [0, y, x2 - new_x, new_y2 - y]

            kpt = kpt - np.stack([new_x, 0])

        else:  # 鼻子在左眼左边
            l_x = (x2 - nose.x) * r_x
            l_y = (y2 - nose.y) * r_y
            new_x2 = min(img_w, int(nose.x + l_x))
            new_y2 = min(img_h, int(nose.y + l_y))
            img_new = img[:new_y2, :new_x2]
            bbox = [x, y, new_x2 - x, new_y2 - y]
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class RetainLeftMouthAndNose(object):
    def __init__(self, start_ratio=0.0, end_ratio=0.5, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug

    def __call__(self, img, bbox, kpt, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        img_h, img_w, c = img.shape
        x2 = x + w
        y2 = y + h
        kpt = np.array(kpt).reshape(-1, 2)
        nose = Point(kpt[2][0], kpt[2][1])
        left_mouth = Point(kpt[3][0], kpt[3][1])
        r_x = self.rng.uniform(self.start_ratio, self.end_ratio)
        r_y = self.rng.uniform(self.start_ratio, self.end_ratio)
        if left_mouth.x < nose.x:  # 鼻子在左嘴右边
            l_x = (x2 - nose.x) * r_x
            l_y = (y2 - nose.y) * r_y
            new_x2 = min(img_w, int(nose.x + l_x))
            new_y = max(0, int(nose.y - l_y))
            img_new = img[new_y:, :new_x2]
            bbox = [x, 0, new_x2 - x, y2 - new_y]

            kpt = kpt - np.stack([0, new_y])

        else:  # 鼻子在左嘴左边
            l_x = (nose.x - x) * r_x
            l_y = (nose.y - y) * r_y
            new_x = max(0, int(nose.x - l_x))
            new_y = max(0, int(nose.y - l_y))
            img_new = img[new_y:, new_x:]
            bbox = [0, 0, x2 - new_x, y2 - new_y]

            kpt = kpt - np.stack([new_x, new_y])
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class RetainRightMouthAndNose(object):
    def __init__(self, start_ratio=0.0, end_ratio=0.5, seed=1234, debug=False):
        assert 0 <= start_ratio < end_ratio <= 1, \
            "The start_ratio and end_ratio need in [0, 1] and start_ratio < end_ratio, but now start_ratio is {}, end_ratio is {}".format(
                start_ratio, end_ratio)
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.rng = np.random.RandomState(seed)
        self.debug = debug

    def __call__(self, img, bbox, kpt, path=None):
        bbox = list(map(int, bbox))
        x, y, w, h = bbox
        x, y, w, h = bbox
        img_h, img_w, c = img.shape
        x2 = x + w
        y2 = y + h
        kpt = np.array(kpt).reshape(-1, 2)
        nose = Point(kpt[2][0], kpt[2][1])
        right_mouth = Point(kpt[4][0], kpt[4][1])
        r_x = self.rng.uniform(self.start_ratio, self.end_ratio)
        r_y = self.rng.uniform(self.start_ratio, self.end_ratio)
        if right_mouth.x > nose.x:  # 鼻子在右嘴左边
            l_x = (nose.x - x) * r_x
            l_y = (nose.y - y) * r_y
            new_x = max(0, int(nose.x - l_x))
            new_y = max(0, int(nose.y - l_y))
            img_new = img[new_y:, new_x:]
            bbox = [0, 0, x2 - new_x, y2 - new_y]

            kpt = kpt - np.stack([new_x, new_y])


        else:  # 鼻子在右嘴右边
            l_x = (x2 - nose.x) * r_x
            l_y = (y2 - nose.y) * r_y
            new_x2 = min(img_w, int(nose.x + l_x))
            new_y = max(0, int(nose.y - l_y))
            img_new = img[new_y:, :new_x2]
            bbox = [x, 0, new_x2 - x, y2 - new_y]

            kpt = kpt - np.stack([0, new_y])
        if self.debug:
            crop_img = draw_box_kpt_with_cwface_label_over_image(img_new, [bbox], kpt)
            plt.imshow(crop_img[:, :, ::-1])
            plt.show()
        return img_new, bbox, kpt, path


class TruncTransformerV2(object):
    def __init__(self, p=0.3, class_weight=None, seed=1234, debug=False):
        self.p = p
        self.rng = np.random.RandomState(seed)
        self.debug = debug
        left = CutLeftFace(start_ratio=0.0, end_ratio=0.4, must_cut=False, seed=seed, debug=debug)
        right = CutRightFace(start_ratio=0.0, end_ratio=0.4, must_cut=False, seed=seed, debug=debug)
        top = CutTopFace(start_ratio=0.0, end_ratio=0.4, must_cut=False, seed=seed, debug=debug)
        bottom = CutBottomFace(start_ratio=0.0, end_ratio=0.4, must_cut=False, seed=seed, debug=debug)
        lefteye = RetainLeftEye(seed=seed, debug=debug)
        righteye = RetainRightEye(seed=seed, debug=debug)
        leftmouth = RetainLeftMouth(seed=seed, debug=debug)
        rightmouth = RetainRightMouth(seed=seed, debug=debug)
        lefteye_leftmouth = RetainLeftEyeAndLeftMouth(seed=seed, debug=debug)
        righteye_rightmouth = RetainRightEyeAndRightMouth(seed=seed, debug=debug)
        lefteye_righteye = RetainLeftEyeAndRightEye(seed=seed, debug=debug)
        leftmouth_rightmouth = RetainLeftMouthAndRightMouth(seed=seed, debug=debug)
        lefteye_nose = RetainLeftEyeAndNose(seed=seed, debug=debug)
        righteye_nose = RetainRightEyeAndNose(seed=seed, debug=debug)
        leftmouth_nose = RetainLeftMouthAndNose(seed=seed, debug=debug)
        rightmouth_nose = RetainRightMouthAndNose(seed=seed, debug=debug)
        self.types = [left, right, top, bottom,
                      lefteye, righteye, leftmouth, rightmouth,
                      lefteye_leftmouth, righteye_rightmouth, lefteye_righteye, leftmouth_rightmouth,
                      lefteye_nose, righteye_nose, leftmouth_nose, rightmouth_nose]
        if class_weight is None:
            class_weight = [1 / len(self.types)] * len(self.types)
            class_weight[-1] = 1 - sum(class_weight[:-1])
        assert abs(sum(class_weight) - 1) <= 0.001
        self.cumsum_weight = np.cumsum(class_weight)

    def sample_class_index_by_weight(self):
        rand_number = self.rng.random() * self.cumsum_weight[-1]
        class_index = min(len(self.cumsum_weight), bisect.bisect_right(self.cumsum_weight, rand_number))
        return class_index

    def __call__(self, img, bbox=None, kpt=None, path=None):
        if bbox is not None:
            bbox = list(map(int, bbox))
        p = self.rng.random()
        if p < self.p:
            s_p = self.rng.random()
            cut_index = self.sample_class_index_by_weight()
            cut = self.types[cut_index]
            img, bbox, kpt = cut(img, bbox, kpt)
        return img, bbox, kpt, path


class TruncTransformer(object):
    def __init__(self, p=0.3, start_ratio=0.0, end_ratio=0.4, num=2, p2=0.2, must_cut_p=0.5, seed=1234,
                 debug=False):
        self.p = p
        self.rng = np.random.RandomState(seed)
        self.debug = debug
        left = CutLeftFace(start_ratio=start_ratio, end_ratio=end_ratio, must_cut=False, seed=seed, debug=debug)
        right = CutRightFace(start_ratio=start_ratio, end_ratio=end_ratio, must_cut=False, seed=seed, debug=debug)
        top = CutTopFace(start_ratio=start_ratio, end_ratio=end_ratio, must_cut=False, seed=seed, debug=debug)
        bottom = CutBottomFace(start_ratio=start_ratio, end_ratio=end_ratio, must_cut=False, seed=seed, debug=debug)
        left_total = CutLeftFace(start_ratio=0.2, end_ratio=0.6, must_cut=True, seed=seed, debug=debug)
        right_total = CutRightFace(start_ratio=0.2, end_ratio=0.6, must_cut=True, seed=seed, debug=debug)
        top_total = CutTopFace(start_ratio=0.2, end_ratio=0.6, must_cut=True, seed=seed, debug=debug)
        bottom_total = CutBottomFace(start_ratio=0.2, end_ratio=0.6, must_cut=True, seed=seed, debug=debug)
        lefteye_leftmouth = RetainLeftEyeAndLeftMouth(seed=seed, debug=debug)
        righteye_rightmouth = RetainRightEyeAndRightMouth(seed=seed, debug=debug)
        lefteye_righteye = RetainLeftEyeAndRightEye(seed=seed, debug=debug)
        leftmouth_rightmouth = RetainLeftMouthAndRightMouth(seed=seed, debug=debug)
        lefteye_nose = RetainLeftEyeAndNose(seed=seed, debug=debug)
        righteye_nose = RetainRightEyeAndNose(seed=seed, debug=debug)
        leftmouth_nose = RetainLeftMouthAndNose(seed=seed, debug=debug)
        rightmouth_nose = RetainRightMouthAndNose(seed=seed, debug=debug)
        self.augs = [lefteye_leftmouth, righteye_rightmouth, lefteye_righteye, leftmouth_rightmouth,
                     lefteye_nose, righteye_nose, leftmouth_nose, rightmouth_nose]
        # self.augs = [rightmouth_nose]
        self.types = [left, right, top, bottom]
        self.severity_types = [left_total, right_total, top_total, bottom_total]
        self.num = num
        self.p2 = p2
        self.must_cut_p = must_cut_p
        from datetime import datetime

        TIMESTAMP = datetime.now().strftime("%y%m%d%H%M%S")
        self.log_path = os.path.join("/dataset/dataset/ssd/kpt_syn/cluster_rst", "trunc_log_" + TIMESTAMP + ".txt")

    def __call__(self, img, bbox, kpt, path=None):
        """kpt的坐标是相对于img的，bbox则是原图上的坐标

        Args:
            img:
            bbox:
            kpt:

        Returns:

        """
        bbox = list(map(int, bbox))
        kpt = np.array(kpt).reshape((-1, 2))
        kpt_x1 = np.min(kpt[:, 0])
        kpt_x2 = np.max(kpt[:, 0])
        kpt_y1 = np.min(kpt[:, 1])
        kpt_y2 = np.max(kpt[:, 1])
        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h
        if kpt_x1 < x or kpt_x2 > x2 or kpt_y1 < y or kpt_y2 > y2:  # 已经有截断的数据则不进行截断
            return img, bbox, kpt, path
        p = self.rng.random()
        if p < self.p:
            s_p = self.rng.random()
            bbox_tmp = np.array(bbox)
            kpt_tmp = np.array(kpt)
            if s_p < 0.05:
                img, bbox, kpt, path, cut_name = self.exec(self.severity_types, bbox, img, kpt, path)
                cut_name = "severity_" + cut_name
            elif 0.05 <= s_p < 0.15:
                aug = self.rng.choice(self.augs)
                cut_name = aug.__class__.__name__
                img, bbox, kpt, path = aug(img, bbox, kpt, path)
            else:
                img, bbox, kpt, path, cut_name = self.exec(self.types, bbox, img, kpt, path)
            # enclose_bbox = get_enclose_bbox_of_kpt(kpt, kpt_num=5)
            # if enclose_bbox[2] >= 1.5 * bbox_tmp[2] or enclose_bbox[3] >= 1.5 * bbox_tmp[3]:
            #     path_sp = os.path.split(path)
            #     save_path = os.path.join("/dataset/dataset/ssd/kpt_syn/", "trunc_imgs", str(cut_name) + "_" + path_sp[1])
            #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #     img_draw = draw_box_kpt_with_cwface_label_over_image(img, [bbox], kpt)
            #     cv2.imwrite(save_path, img_draw)
            #     os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            #     with open(self.log_path, "a+") as fw:
            #         new_line = path + " 0 1 " + " ".join(
            #             list(map(str, bbox_tmp))) + " " + \
            #                    " ".join(list(map(str, kpt_tmp.reshape(-1).tolist()))) + "\n"
            #         fw.write(new_line)
            #         new_line = save_path + " 0 1 " + " ".join(
            #             list(map(str, bbox))) + " " + \
            #                    " ".join(list(map(str, kpt.reshape(-1).tolist()))) + "\n"
            #         fw.write(new_line)
        return img, bbox, kpt, path

    def exec(self, type, bbox, img, kpt, path=None, num=2, p2=0.2, must_cut_p=0.5):
        cut = self.rng.choice(type)
        cut_name = cut.__class__.__name__
        tmp_cut = cut.must_cut
        must_p = self.rng.random()
        if must_p < self.must_cut_p:
            cut.must_cut = True
        img, bbox, kpt, path = cut(img, bbox, kpt, path)
        cut.must_cut = tmp_cut
        if num > 1:  # 进行多种组合裁剪
            for i in range(num - 1):
                p = self.rng.random()
                if p < p2:
                    cut = self.rng.choice(type)
                    cut_name += cut.__class__.__name__
                    tmp_cut = cut.must_cut
                    must_p = self.rng.random()
                    if must_p < must_cut_p:
                        cut.must_cut = True
                    img, bbox, kpt, path = cut(img, bbox, kpt, path)
                    cut.must_cut = tmp_cut
        return img, bbox, kpt, path, cut_name


if __name__ == '__main__':
    path = "3.jpg"
    img = cv2.imread(path)
    bbox = [15, 22, 88, 95]
    kpt = [44, 65, 87, 63, 67, 89, 52, 102, 81, 100]
    # kpt = [44, 65, 87, 63, 100, 89, 52, 102, 81, 100]
    draw_img = draw_box_kpt_with_cwface_label_over_image(img, [bbox], kpt)
    plt.imshow(draw_img[:, :, ::-1])
    plt.show()
    # a = TruncTransformer(p=0.3, start_ratio=0.0, end_ratio=0.4, num=2, p2=0.2, must_cut_p=0.5, seed=1234, debug=True)
    class_weight = [
        0.2, 0.2, 0.2, 0.2,
        0.01, 0.01, 0.01, 0.01,
        0.03, 0.03, 0.03, 0.03,
        0.01, 0.01, 0.01, 0.01,
    ]
    a = TruncTransformer(p=0.3, seed=1234, debug=True)
    # a = RetainRightMouthAndNose(seed=1234, debug=True)
    b = a(img, bbox, kpt, path)
    b = a(img, bbox, kpt, path)
    b = a(img, bbox, kpt, path)
    print("END")
