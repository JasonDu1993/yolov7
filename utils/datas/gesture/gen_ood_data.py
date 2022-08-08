# -*- coding: utf-8 -*-
# @Time    : 2022/7/4 14:09
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : gen_cifar10.py
# @Software: PyCharm
import os
import cv2
import numpy as np

np.random.seed(1234)


def rand_bbox(box):
    """

    Args:
        box: a list, value is [x, y, w, h]
        lam:

    Returns:

    """
    p = np.random.random()
    if p <= 0.5:
        return box
    ratio_w = float(np.random.uniform(0.9, 1))
    ratio_h = float(np.random.uniform(0.9, 1))
    x, y, w, h = box
    new_w = int(w * ratio_w)
    new_h = int(h * ratio_h)
    new_x = max(int(x - (new_w - w) * 0.5), 0)
    new_y = max(int(y - (new_h - h) * 0.5), 0)
    bbox = [new_x, new_y, new_w, new_h]
    return bbox


def gen_ood(txt_path, save_path, root=None):
    with open(txt_path, "r", encoding="utf-8") as fr:
        with open(save_path, "w", encoding="utf-8") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                img_path = line_sp[0] if root is None else os.path.join(root, line_sp[0])
                label = line_sp[1]
                label = str(-1)
                img = cv2.imread(img_path)
                img_h, img_w, img_c = img.shape
                box = rand_bbox([0, 0, img_w, img_h])
                box_str = " ".join(list(map(str, box)))
                new_line = img_path + " " + label + " " + str(img_w) + " " + str(img_h) + " " + box_str + "\n"
                print(new_line)
                fw.write(new_line)


if __name__ == '__main__':
    datas = {
        "cifar10": {
            "txt_path": "/dataset/dataset/cifar10/test/disk.list",
            "save_path": "/dataset/dataset/cifar10/test/test.cifar10.box.txt"
        },
        "imagenet": {
            "txt_path": "/dataset/dataset/ssd/gesture/imagenet/imagenet_ood.txt",
            "save_path": "/dataset/dataset/ssd/gesture/imagenet/imagenet_ood.box.txt",
            "root": "/zhouyafei/image_recog_data/ImageNet/ILSVRC2015/Data/CLS-LOC/"
        }
    }
    data_name = ["imagenet"]
    for data_name in data_name:
        print("data name:{}".format(data_name))
        txt_path = datas[data_name]["txt_path"]
        save_path = datas[data_name]["save_path"]
        root = datas[data_name]["root"] if "root" in datas[data_name] else None
        gen_ood(txt_path, save_path, root)
