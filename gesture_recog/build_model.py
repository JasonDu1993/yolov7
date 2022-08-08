# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 14:11
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : gesture_model.py
# @Software: PyCharm
import os
import sys
import numpy as np
from importlib.machinery import SourceFileLoader
from importlib import import_module
from gesture_recog.model import Model, load_weight
from gesture_recog.transformer.crop_and_padding import CropAndPaddingTransformer
from gesture_recog.transformer.reshape import ReshapeTransformer
from gesture_recog.transformer.normalize import NormalizeTransformer
from gesture_recog.transformer.totensor import ToTensor
from gesture_recog.transformer.compose import Compose


def get_test_transformer(config):
    opt = SourceFileLoader('module.name', config).load_module().opt
    trans = []
    kpt_type = opt.data.kpt_type
    assert kpt_type in ["regression", "heatmap"]
    if "crop_padding" in opt.data:
        crop_padding = opt.data.crop_padding
    else:
        crop_padding = opt.data.trans.crop_padding
    trans.append(
        CropAndPaddingTransformer(output_shape=opt.data.input_shape, scale_box=crop_padding.scale_box,
                                  add_padding=crop_padding.add_padding,
                                  keep_ratio=crop_padding.keep_ratio, seed=crop_padding.seed,
                                  debug=False))
    trans.append(ReshapeTransformer(opt.data.input_shape, kpt_type=kpt_type))
    trans.append(NormalizeTransformer(bias=opt.data.bias, scale=opt.data.scale, kpt_type=kpt_type))
    trans.append(ToTensor())
    return Compose(trans)


def build_model(config, model, weight_path):
    # import model
    opt = SourceFileLoader('module.name', config).load_module().opt

    modual_name = '.'.join(model.split('.')[:-1])
    class_name = model.split('.')[-1]
    Model = getattr(import_module(modual_name), class_name)
    if "garbage_class_num" in opt.data:
        garbage_class_num = opt.data.garbage_class_num
    else:
        garbage_class_num = 0
    model = Model(opt.data.input_shape, opt.data.class_num, garbage_class_num=garbage_class_num, deploy=True)
    load_weight(model, weight_path)
    # model = model.to(args["gpu"])
    model.eval()  # 不加这句话预测是错误的
    return model


def get_gesture_inputs(imgs, trans, bbox):
    kpts = None
    img_path = None
    new_imgs = []
    for img in imgs:
        img_new, bbox_new, kpts, img_path = trans(img, bbox, kpts, img_path)
        new_imgs.append(img_new)
    new_imgs = np.concatenate(new_imgs, axis=0)
    return new_imgs


if __name__ == '__main__':
    config = "tools/facekpt/core/models/exp220107b/config.py"
    model = "gesture_recog.model.Model"
    weight_path = "/zhoudu/checkpoints/gesture_models/gesture220617a_115510/checkpoint/kpt-epoch109.weights"
    gpu = 1
