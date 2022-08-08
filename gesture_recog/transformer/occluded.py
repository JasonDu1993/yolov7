# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 11:52
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : rotate.py
# @Software: PyCharm
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import bisect
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.draw_box_kpt_utils import draw_box_kpt_with_cwface_label_over_image
from tools.facekpt.utils.box_and_kpt_utils import get_enclose_bbox_of_kpt
from tools.facekpt.utils.parse_line import parse_cwface_label_path
from tools.facekpt.transformer.crop_and_padding import CropAndPaddingTransformer
from tools.facekpt.transformer.trunc import RetainRightMouthAndNose


class LeftEyeOccTransformer(object):
    def __init__(self, p=0.3, seed=1234, debug=False):
        self.rng = np.random.RandomState(seed)
        self.p = p
        self.debug = debug

    def __call__(self, img, bbox=None, kpt=None):
        """kpt的坐标是相对于img的，bbox则是原图上的坐标

        Args:
            img:
            bbox:
            kpt:

        Returns:

        """
        img_h, img_w, c = img.shape
        if bbox is not None:
            bbox = list(map(int, bbox))
            x, y, w, h = bbox
        else:
            x, y, w, h = 0, 0, img_w, img_h

        if self.debug:
            img_draw = draw_box_kpt_with_cwface_label_over_image(img, [bbox], kpt, kpt_num=5)
            plt.imshow(img_draw[:, :, ::-1])
            plt.show()
        return img, bbox, kpt, path


def crop_corner(img, crop_size):
    x1, y1, x2, y2 = list(map(int, crop_size))
    img = img[y1:y2, x1:x2]
    return img


def paste_corner(img, p_img):
    p_h, p_w, p_c = p_img.shape
    # 左上角
    p_x1 = 0
    p_y1 = 0
    p_x2 = p_x1 + p_w
    p_y2 = p_y1 + p_h
    img[p_y1:p_y2, p_x1:p_x2] = p_img
    return img


def paste_img(src_path, src_box, src_kpt, dst_path, dst_box, dst_kpt, ):
    dst_x, dst_y, dst_w, dst_h = dst_box
    if dst_w < 40 and dst_h < 40:
        return
    src_img = cv2.imread(src_path)
    dst_img = cv2.imread(dst_path)

    src_img_draw = draw_box_kpt_with_cwface_label_over_image(src_img, [src_box], src_kpt, radius=1, box_thick=1)
    plt.imshow(src_img_draw[:, :, ::-1])
    plt.show()

    rng = np.random.RandomState(1234)
    # ratio_paste_w = rng.uniform(0.15, 0.4)
    # ratio_paste_h = rng.uniform(0.15, 0.4)
    # ratio_paste_w = rng.uniform(0.25, 0.45)
    # ratio_paste_h = rng.uniform(0.25, 0.45)
    ratio_paste_w = 0.5
    ratio_paste_h = 0.5
    crop = CropAndPaddingTransformer((3, 112, 96), keep_ratio=True)
    dst_img_new, dst_box_new, dst_kpt_new, dst_path = crop(dst_img, dst_box, dst_kpt, dst_path)
    dst_img_draw = draw_box_kpt_with_cwface_label_over_image(dst_img, [dst_box_new], dst_kpt, radius=1, box_thick=1)
    plt.imshow(dst_img_draw[:, :, ::-1])
    plt.show()

    dst_h, dst_w, dst_c = dst_img_new.shape
    dst_x, dst_y, dst_w, dst_h = dst_box_new
    paste_w = int(ratio_paste_w * dst_w * 0.8)
    paste_h = int(ratio_paste_h * dst_h * 0.8)

    src_x, src_y, src_w, src_h = src_box
    src_x2 = src_x + src_w
    src_y2 = src_y + src_h
    src_crop_w = int(ratio_paste_w * src_w)
    src_crop_h = int(ratio_paste_h * src_h)
    crop_size = [src_x2 - src_crop_w, src_y2 - src_crop_h, src_x2, src_y2]
    p_img = crop_corner(src_img, crop_size)
    plt.imshow(p_img[:, :, ::-1])
    plt.show()
    plt.imshow(dst_img_new[:, :, ::-1])
    plt.show()
    p_img = cv2.resize(p_img, (paste_w, paste_h))
    new_dst_img = paste_corner(dst_img_new, p_img)
    plt.imshow(new_dst_img[:, :, ::-1])
    plt.show()


def demo():
    # Standard imports
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Read images
    src = cv2.imread("1.png")
    dst = cv2.imread("2.png")
    dst_h, dst_w, dst_c = dst.shape
    # Create a rough mask around the airplane.
    src = cv2.resize(src, (int(dst_w * 0.4), int(dst_h * 0.4)))
    plt.imshow(src[:, :, ::-1])
    plt.show()
    plt.imshow(dst[:, :, ::-1])
    plt.show()

    # 方法一
    src_mask = np.zeros(src.shape, src.dtype)
    # 当然我们比较懒得话，就不需要下面两行，只是效果差一点。
    # 不使用的话我们得将上面一行改为 mask = 255 * np.ones(obj.shape, obj.dtype) <-- 全白
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # poly = np.array([[4, 80], [30, 54], [151, 63], [254, 37], [298, 90], [272, 134], [43, 122]], np.int32)
    poly = np.concatenate(contours).reshape((-1, 2))
    cv2.fillPoly(src_mask, [poly], (255, 255, 255))

    # 方法二
    # src_mask = 255 * np.ones(src.shape, src.dtype)
    # 这是 飞机 CENTER 所在的地方
    center = (800, 100)
    center = (int(dst_w * 0.2), int(dst_h * 0.2))

    # Clone seamlessly.
    output = cv2.seamlessClone(src, dst, src_mask, center, cv2.MIXED_CLONE)  # MIXED_CLONE NORMAL_CLONE

    # 保存结果
    # cv2.imwrite("images/opencv-seamless-cloning-example.jpg", output)
    plt.imshow(output[:, :, ::-1])
    plt.show()


if __name__ == '__main__':
    path = "t.txt"
    img_path_list, boxes_list, kpts_list, faceid_list = parse_cwface_label_path(path)
    paste_img(img_path_list[0], boxes_list[0], kpts_list[0], img_path_list[1], boxes_list[1], kpts_list[1])
    # path = "3.jpg"
    # img = cv2.imread(path)
    # bbox = [15, 22, 88, 95]
    # # bbox = [10, 18, 93, 100]  # 逆时针旋转45
    # # bbox = [10, 24, 87, 94]  # 逆时针旋转20
    # # bbox = [25, 31, 92, 87]  # 顺时针时针旋转45
    # x, y, w, h = bbox
    # center_x = x + w // 2
    # center_y = y + h // 2
    # center = (center_x, center_y)
    # center = np.array(center).reshape(-1, 2)
    # kpt = [44, 65, 87, 63, 67, 89, 52, 102, 81, 100]
    # kpt_new = np.concatenate([np.array(kpt).reshape(-1, 2), center])
    # img_draw = draw_box_kpt_with_cwface_label_over_image(img, [bbox], kpt_new, kpt_num=6)
    # plt.imshow(img_draw[:, :, ::-1])
    # plt.show()
    # t = LeftEyeOccTransformer(p=1, debug=True)
    # img, bbox, kpt, path = t(img, bbox, kpt, path)
    # print("END")
