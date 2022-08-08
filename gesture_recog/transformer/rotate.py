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


class RotateTransformer(object):
    def __init__(self, min_angle=-30, max_angle=30, fixed_angle=None, seed=1234, debug=False):
        self.rng = np.random.RandomState(seed)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.debug = debug
        self.fixed_angle = fixed_angle

    def __call__(self, img, bbox=None, kpt=None):
        """kpt的坐标是相对于img的，bbox则是原图上的坐标

        Args:
            img:
            bbox:
            kpt:

        Returns:

        """
        if self.fixed_angle is None:
            rot_angle = self.rng.uniform(low=self.min_angle, high=self.max_angle)
        else:
            rot_angle = self.fixed_angle
        img_h, img_w, c = img.shape
        if bbox is not None:
            bbox = list(map(int, bbox))
            x, y, w, h = bbox
        else:
            x, y, w, h = 0, 0, img_w, img_h
        center_x = x + w // 2
        center_y = y + h // 2
        center = (center_x, center_y)
        # 第一个参数旋转中心，第二个参数旋转角度,正值表示逆时针旋转，第三个参数：缩放比例
        M = cv2.getRotationMatrix2D(center, rot_angle, 1)
        # 第三个参数：变换后的图像大小
        img = cv2.warpAffine(img, M, (img_w, img_h))
        # cv2.imwrite("3_r_neg45.jpg", img)

        center = np.array(center).reshape(-1, 2)
        if bbox is not None:
            bbox_p = np.array([[x, y + h // 2], [x + w, y + h // 2], [x + w // 2, y], [x + w // 2, y + h]])
            bbox_p_new = (bbox_p - center).dot(M[:2, :2].T) + center
            new_x = int(np.min(bbox_p_new[:, 0]))
            new_x2 = int(np.max(bbox_p_new[:, 0]))
            new_y = int(np.min(bbox_p_new[:, 1]))
            new_y2 = int(np.max(bbox_p_new[:, 1]))
            bbox = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
        if kpt is not None:
            kpt = np.array(kpt)
            kpt = kpt.reshape([-1, 2])
            kpt = (kpt - center).dot(M[:2, :2].T) + center
        if self.debug:
            if kpt is not None:
                kpt_new = np.concatenate([kpt, center])
            else:
                kpt_new = None
            img_draw = draw_box_kpt_with_cwface_label_over_image(img, [bbox], kpt_new, kpt_num=6)
            plt.imshow(img_draw[:, :, ::-1])
            plt.show()
        return img, bbox, kpt


def exchange(kpt, s, d):
    a = copy.deepcopy(kpt[s, :])
    kpt[s, :] = kpt[d, :]
    kpt[d, :] = a
    return kpt


class HorizontalFlipTransformer(object):
    def __init__(self, debug=False):
        self.debug = debug

    def __call__(self, img, bbox=None, kpt=None):
        # --image 输入图像
        # --flipCode 1沿y轴水平翻转，0沿x轴垂直翻转，-1水平垂直翻转
        img = cv2.flip(img, 1)  # 水平翻转
        img_h, img_w, c = img.shape
        if bbox is not None:
            x, y, w, h = bbox
            x2 = x + w
            y2 = y + h
            new_x = img_w - x2
            new_x2 = img_w - x
            bbox = [new_x, y, w, h]
        if kpt is not None:
            kpt = np.array(kpt)
            kpt = kpt.reshape([-1, 2])
            kpt[:, 0] = img_w - kpt[:, 0]
            kpt = exchange(kpt, 0, 1)
            kpt = exchange(kpt, 3, 4)
        if self.debug:
            img_draw = draw_box_kpt_with_cwface_label_over_image(img, [bbox], kpt)
            plt.imshow(img_draw[:, :, ::-1])
            plt.show()
        return img, bbox, kpt


class VerticalFlipTransformer(object):
    def __init__(self, debug=False):
        self.debug = debug

    def __call__(self, img, bbox=None, kpt=None):
        # --image 输入图像
        # --flipCode 1沿y轴水平翻转，0沿x轴垂直翻转，-1水平垂直翻转
        img = cv2.flip(img, 0)  # 垂直翻转
        img_h, img_w, c = img.shape
        if bbox is not None:
            x, y, w, h = bbox
            x2 = x + w
            y2 = y + h
            new_y = img_h - y2
            new_y2 = img_h - y
            bbox = [x, new_y, w, h]
        if kpt is not None:
            kpt = np.array(kpt)
            kpt = kpt.reshape([-1, 2])
            kpt[:, 1] = img_h - kpt[:, 1]
            kpt = exchange(kpt, 0, 1)
            kpt = exchange(kpt, 3, 4)
        if self.debug:
            img_draw = draw_box_kpt_with_cwface_label_over_image(img, [bbox], kpt)
            plt.imshow(img_draw[:, :, ::-1])
            plt.show()
        return img, bbox, kpt


class RotateCombination(object):
    def __init__(self, p=0.2, min_angle=-30, max_angle=30, class_weight=None, seed=1234, debug=False):
        assert min_angle <= 0
        assert max_angle >= 0
        self.rng = np.random.RandomState(seed)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.debug = debug
        self.p = p
        vertical_flip = VerticalFlipTransformer(debug=debug)
        horizontal_flip = HorizontalFlipTransformer(debug=debug)
        rotate_clockwise_45 = RotateTransformer(fixed_angle=-45, debug=debug)  # 顺时针旋转
        rotate_counterclockwise_45 = RotateTransformer(fixed_angle=45, debug=debug)  # 逆时针旋转
        rotate_clockwise_90 = RotateTransformer(fixed_angle=-90, debug=debug)
        rotate_counterclockwise_90 = RotateTransformer(fixed_angle=90, debug=debug)
        rotate_180 = RotateTransformer(fixed_angle=180, debug=debug)
        rotate_clockwise_random_angle = RotateTransformer(min_angle=min_angle, max_angle=0, debug=debug)
        rotate_counterclockwise_random_angle = RotateTransformer(min_angle=0, max_angle=max_angle, debug=debug)
        self.types = [rotate_clockwise_45, rotate_counterclockwise_45,
                      rotate_clockwise_90, rotate_counterclockwise_90,
                      rotate_180, vertical_flip,
                      rotate_clockwise_random_angle, rotate_counterclockwise_random_angle]
        if class_weight is None:
            # class_weight = [1 / len(self.types)] * len(self.types)
            # class_weight[-1] = 1 - sum(class_weight[:-1])
            class_weight = [0.1, 0.1,
                            0.05, 0.05,
                            0.05, 0.05,
                            0.3, 0.3]
        assert abs(sum(class_weight) - 1) <= 0.001
        self.cumsum_weight = np.cumsum(class_weight)

        from datetime import datetime

        TIMESTAMP = datetime.now().strftime("%y%m%d%H%M%S")
        self.log_path = os.path.join("/dataset/dataset/ssd/kpt_syn/cluster_rst", "rotate_log_" + TIMESTAMP + ".txt")

    def sample_class_index_by_weight(self):
        rand_number = self.rng.random() * self.cumsum_weight[-1]
        class_index = min(len(self.cumsum_weight), bisect.bisect_right(self.cumsum_weight, rand_number))
        return class_index

    def __call__(self, img, bbox=None, kpt=None, path=None):
        p = self.rng.random()
        if p < self.p:
            index = self.sample_class_index_by_weight()
            rot = self.types[index]
            bbox_tmp = np.array(bbox)
            kpt_tmp = np.array(kpt)
            img, bbox, kpt = rot(img, bbox, kpt)
            # enclose_bbox = get_enclose_bbox_of_kpt(kpt, kpt_num=5)
            # if enclose_bbox[2] >= 1.5 * bbox_tmp[2] or enclose_bbox[3] >= 1.5 * bbox_tmp[3]:
            #     path_sp = os.path.split(path)
            #     save_path = os.path.join("/dataset/dataset/ssd/kpt_syn/", "rotate_imgs", str(index) + "_" + path_sp[1])
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


if __name__ == '__main__':
    path = "3.jpg"
    img = cv2.imread(path)
    bbox = [15, 22, 88, 95]
    # bbox = [10, 18, 93, 100]  # 逆时针旋转45
    # bbox = [10, 24, 87, 94]  # 逆时针旋转20
    # bbox = [25, 31, 92, 87]  # 顺时针时针旋转45
    x, y, w, h = bbox
    center_x = x + w // 2
    center_y = y + h // 2
    center = (center_x, center_y)
    center = np.array(center).reshape(-1, 2)
    kpt = [44, 65, 87, 63, 67, 89, 52, 102, 81, 100]
    kpt_new = np.concatenate([np.array(kpt).reshape(-1, 2), center])
    img_draw = draw_box_kpt_with_cwface_label_over_image(img, [bbox], kpt_new, kpt_num=6)
    plt.imshow(img_draw[:, :, ::-1])
    plt.show()
    t = RotateCombination(p=1, debug=True)
    img, bbox, kpt, path = t(img, bbox, kpt, path)
    print("END")
