# -*- coding: utf-8 -*-
# @Time    : 2021/10/21 15:43
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : normalize.py
# @Software: PyCharm
import os, sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))


class NormalizeTransformer(object):
    def __init__(self, bias, scale, kpt_type="regression"):
        self.bias = bias
        self.scale = scale
        self.kpt_type = kpt_type
        from datetime import datetime

        TIMESTAMP = datetime.now().strftime("%y%m%d%H%M%S")
        self.log_path = os.path.join("/dataset/dataset/ssd/kpt_syn/cluster_rst", "normalize_log_" + TIMESTAMP + ".txt")

    def __call__(self, img, bbox, kpt=None, path=None):
        """kpt的坐标是相对于img的，bbox则是原图上的坐标

        Args:
            img:
            bbox:
            kpt:

        Returns:

        """
        img_tmp = img.copy()
        img = img.astype(np.float32)
        img = (img - self.bias) * self.scale
        if kpt is not None:
            bbox_tmp = np.array(bbox)
            kpt_tmp = np.array(kpt)
            if self.kpt_type == "regression":
                h, w, c = img.shape
                # 该坐标是以img为标准的坐标，直接除以图片的宽高，如果能保证前面剪切的时候是按照bbox进行crop的则可以使用bbox的宽高
                kpt = np.reshape(kpt, (-1, 2))  # shape [5, 2]
                kpt = kpt / np.stack([w, h])
                kpt = kpt.astype(np.float32)
            elif self.kpt_type == "heatmap":
                kpt = kpt
            # if np.sum(kpt <= -0.5) or np.sum(kpt >= 1.5):
            #     path_sp = os.path.split(path)
            #     save_path = os.path.join("/dataset/dataset/ssd/kpt_syn/", "normalize_imgs",
            #                              str("reshape") + "_" + path_sp[1])
            #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #     img_draw = draw_box_kpt_with_cwface_label_over_image(img_tmp, [bbox], kpt_tmp, draw_box=False)
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
