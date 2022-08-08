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
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.draw_box_kpt_utils import draw_box_kpt_with_cwface_label_over_image

kpt_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
             (0, 255, 255)]  # left_eye(green), right_eye(red), nose(blue), left_mouth(cyan), right_mouth(yellow)


class CropAndPaddingTransformer(object):
    def __init__(self, output_shape=None, scale_box=1.0, add_padding=False, keep_ratio=True, seed=1234, debug=False):
        if add_padding and keep_ratio:
            raise Exception("add_padding and keep_ratio can't both be true")
        self.rng = np.random.RandomState(seed)
        self.add_padding = add_padding
        if not output_shape:
            self.add_padding = False
        if debug:
            print("output_shape is {}, add_padding is {}".format(output_shape, self.add_padding))
        self.c, self.out_h, self.out_w = output_shape
        self.debug = debug
        self.keep_ratio = keep_ratio
        self.scale_box = scale_box
        from datetime import datetime

        TIMESTAMP = datetime.now().strftime("%y%m%d%H%M%S")
        self.log_path = os.path.join("/dataset/dataset/ssd/kpt_syn/cluster_rst", "crop_log_" + TIMESTAMP + ".txt")

    def __call__(self, img, bbox, kpt=None, path=None):
        """kpt的坐标是相对于img的，bbox则是原图上的坐标

        Args:
            img:
            bbox:
            kpt:

        Returns:

        """
        h, w, c = img.shape
        x0, y0, bbox_w, bbox_h = list(map(int, bbox))
        x1 = x0 + bbox_w
        y1 = y0 + bbox_h
        bbox_tmp = np.array(bbox)
        kpt_tmp = np.array(kpt)
        if self.keep_ratio:
            ratio = max(bbox_w * self.scale_box / self.out_w, bbox_h * self.scale_box / self.out_h)
            new_bbox_w = int(self.out_w * ratio)
            new_bbox_h = int(self.out_h * ratio)
            x0 = x0 - (new_bbox_w - bbox_w) // 2
            y0 = y0 - (new_bbox_h - bbox_h) // 2
            bbox_w = new_bbox_w
            bbox_h = new_bbox_h
            x1 = x0 + bbox_w
            y1 = y0 + bbox_h
            bbox = [x0, y0, bbox_w, bbox_h]  # 扩展之后原图上的坐标，可能为负值

        top = np.abs(np.minimum(0, y0))
        bottom = np.abs(h - np.maximum(h, y1))
        left = np.abs(np.minimum(0, x0))
        right = np.abs(w - np.maximum(w, x1))
        # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        new_x0 = x0 + left
        new_y0 = y0 + top
        new_x1 = new_x0 + bbox_w
        new_y1 = new_y0 + bbox_h
        img = img[new_y0:new_y1, new_x0:new_x1, :]

        # x0 = np.maximum(0, x0)
        # y0 = np.maximum(0, y0)
        # x1 = np.minimum(w, x1)
        # y1 = np.minimum(h, y1)
        # img = img[y0:y1, x0:x1, :]
        # bbox_w = x1 - x0
        # bbox_h = y1 - y0
        if self.add_padding:
            ratio = max(bbox_w * self.scale_box / self.out_w, bbox_h * self.scale_box / self.out_h)
            new_bbox_h = math.ceil(self.out_h * ratio)
            new_bbox_w = math.ceil(self.out_w * ratio)
            pad_h = new_bbox_h - bbox_h
            pad_w = new_bbox_w - bbox_w
            top = int(pad_h / 2)
            bottom = pad_h - top
            left = int(pad_w / 2)
            right = pad_w - left
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
            x0 = x0 - left
            y0 = y0 - top
            bbox = [x0, y0, new_bbox_w, new_bbox_h]
        if kpt is not None:
            kpt = np.array(kpt)
            kpt = np.reshape(kpt, (-1, 2))  # shape [5, 2]
            kpt = kpt - np.stack([x0, y0])
            # 注意，这里输出的bbox是原图上扩展之后的坐标，目的是为了存储框信息时知道原图上的框的位置
            # enclose_bbox = get_enclose_bbox_of_kpt(kpt, kpt_num=5)
            # if enclose_bbox[2] >= 1.5 * bbox_tmp[2] or enclose_bbox[3] >= 1.5 * bbox_tmp[3]:
            #     path_sp = os.path.split(path)
            #     save_path = os.path.join("/dataset/dataset/ssd/kpt_syn/", "crop_imgs", str("crop") + "_" + path_sp[1])
            #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #     img_draw = draw_box_kpt_with_cwface_label_over_image(img, [bbox], kpt, draw_box=False)
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
        if self.debug:
            img_draw = draw_box_kpt_with_cwface_label_over_image(img, [[0, 0, bbox[2], bbox[3]]], kpt)
            plt.imshow(img_draw[:, :, ::-1])
            plt.show()
        return img, bbox, kpt, path


if __name__ == '__main__':
    img = cv2.imread("3.jpg")
    t = CropAndPaddingTransformer(output_shape=(3, 100, 100), debug=True)
    bbox = [10, -5, 100, 150]
    kpt = [[10, 2, 100, 2, 50, 50, 110, 10, 120, 90]]
    kpt = np.array(kpt).reshape(5, 2)
    for i, pt in enumerate(kpt):
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, kpt_color[i], -1)
    plt.imshow(img[:, :, ::-1])
    plt.show()
    m = t(img, bbox, kpt)
