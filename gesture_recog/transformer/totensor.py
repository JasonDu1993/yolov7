# -*- coding: utf-8 -*-
# @Time    : 2021/10/21 15:28
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : totensor.py
# @Software: PyCharm
import cv2
import torch
import numpy as np


class ToTensor(object):
    def __call__(self, img, bbox=None, kpt=None, path=None):
        if not isinstance(img, torch.Tensor):
            if img.ndim == 2:
                img = img[np.newaxis, :]
            elif img.ndim == 3:
                img = img.transpose((2, 0, 1))
            else:
                raise ValueError('Invalid image dimensions.')
            if bbox is not None:
                if isinstance(bbox, list):
                    bbox = np.array(bbox)
                bbox = torch.from_numpy(bbox).contiguous()
            if kpt is not None:
                kpt = torch.from_numpy(kpt).contiguous().float()
            return torch.from_numpy(img).contiguous(), bbox, kpt, path
        else:
            return img.contiguous(), bbox, kpt, path
