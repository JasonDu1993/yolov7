# -*- coding: utf-8 -*-
# @Time    : 2022/7/22 11:31
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : deploy.py
# @Software: PyCharm
import os
from copy import deepcopy
from models.yolo import Model
import torch
import yaml
from utils.torch_utils import select_device, is_parallel


def yolov7_reparameterization():
    """YOLOv7 reparameterization
    Returns:

    """
    device = select_device('0', batch_size=1)
    # model trained by cfg/training/*.yaml
    weight_path = "/zhoudu/checkpoints/gesture/yolov7/yolov7_416_jsc/weights/best.pt"
    modef_config = "cfg/deploy/yolov7_416.yaml"
    save_path_sp = os.path.split(weight_path)
    save_path = os.path.join(save_path_sp[0], "deploy." + save_path_sp[1])
    print("src_path:{}".format(weight_path))
    print("save_path:{}".format(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = torch.load(weight_path, map_location=device)
    # ckpt = torch.load('/dataset/dataset/ssd/model_meta/yolov7/yolov7.pt', map_location=device)
    # reparameterized model in cfg/deploy/*.yaml

    nc = 1
    model = Model(modef_config, ch=3, nc=nc).to(device)
    print("model.state_dict()", model.state_dict().keys())
    # copy intersect weights
    state_dict = ckpt['model'].float().state_dict()
    exclude = []
    intersect_state_dict = {k: v for k, v in state_dict.items() if
                            k in model.state_dict() and not any(x in k for x in exclude) and v.shape ==
                            model.state_dict()[
                                k].shape}
    model.load_state_dict(intersect_state_dict, strict=False)
    model.names = ckpt['model'].names
    # model.nc = ckpt['model'].nc
    model.nc = nc
    with open('cfg/deploy/yolov7.yaml') as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    anchors = len(yml['anchors'])
    idx = 105
    # reparametrized YOLOR
    for i in range((model.nc + 5) * anchors):
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= \
            state_dict['model.{}.im.0.implicit'.format(idx)].data[:, i, ::].squeeze()
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= \
            state_dict['model.{}.im.1.implicit'.format(idx)].data[:, i, ::].squeeze()
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= \
            state_dict['model.{}.im.2.implicit'.format(idx)].data[:, i, ::].squeeze()
    model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx)].mul(
        state_dict['model.{}.ia.0.implicit'.format(idx)]).sum(1).squeeze()
    model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx)].mul(
        state_dict['model.{}.ia.1.implicit'.format(idx)]).sum(1).squeeze()
    model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx)].mul(
        state_dict['model.{}.ia.2.implicit'.format(idx)]).sum(1).squeeze()
    model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict[
        'model.{}.im.0.implicit'.format(idx)].data.squeeze()
    model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict[
        'model.{}.im.1.implicit'.format(idx)].data.squeeze()
    model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict[
        'model.{}.im.2.implicit'.format(idx)].data.squeeze()

    # model to be saved
    ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'optimizer': None,
            'training_results': None,
            'epoch': -1}

    # save reparameterized model
    torch.save(ckpt, save_path)


def yolov7_tiny_reparameterization():
    """YOLOv7 tiny reparameterization
    Returns:

    """
    device = select_device('0', batch_size=1)
    # model trained by cfg/training/*.yaml
    weight_path = "/zhoudu/checkpoints/gesture/yolov7/yolov7_tiny_jsc/weights/best.pt"
    modef_config = "cfg/deploy/yolov7-tiny.yaml"
    save_path_sp = os.path.split(weight_path)
    save_path = os.path.join(save_path_sp[0], "deploy." + save_path_sp[1])
    print("src_path:{}".format(weight_path))
    print("save_path:{}".format(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = torch.load(weight_path, map_location=device)
    # ckpt = torch.load('/dataset/dataset/ssd/model_meta/yolov7/yolov7.pt', map_location=device)
    # reparameterized model in cfg/deploy/*.yaml

    nc = 1
    model = Model(modef_config, ch=3, nc=nc).to(device)
    print("model.state_dict()", model.state_dict().keys())
    # copy intersect weights
    state_dict = ckpt['model'].float().state_dict()
    exclude = []
    intersect_state_dict = {k: v for k, v in state_dict.items() if
                            k in model.state_dict() and not any(x in k for x in exclude) and v.shape ==
                            model.state_dict()[
                                k].shape}
    model.load_state_dict(intersect_state_dict, strict=False)
    model.names = ckpt['model'].names
    model.nc = nc
    with open('cfg/deploy/yolov7.yaml') as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    anchors = len(yml['anchors'])
    idx = 77
    # reparametrized YOLOR
    for i in range((model.nc + 5) * anchors):
        model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= \
            state_dict['model.{}.im.0.implicit'.format(idx)].data[:, i, ::].squeeze()
        model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= \
            state_dict['model.{}.im.1.implicit'.format(idx)].data[:, i, ::].squeeze()
        model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= \
            state_dict['model.{}.im.2.implicit'.format(idx)].data[:, i, ::].squeeze()
    model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx)].mul(
        state_dict['model.{}.ia.0.implicit'.format(idx)]).sum(1).squeeze()
    model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx)].mul(
        state_dict['model.{}.ia.1.implicit'.format(idx)]).sum(1).squeeze()
    model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx)].mul(
        state_dict['model.{}.ia.2.implicit'.format(idx)]).sum(1).squeeze()
    model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict[
        'model.{}.im.0.implicit'.format(idx)].data.squeeze()
    model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict[
        'model.{}.im.1.implicit'.format(idx)].data.squeeze()
    model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict[
        'model.{}.im.2.implicit'.format(idx)].data.squeeze()

    # model to be saved
    ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'optimizer': None,
            'training_results': None,
            'epoch': -1}

    # save reparameterized model
    torch.save(ckpt, save_path)


if __name__ == '__main__':
    # yolov7_reparameterization()
    yolov7_tiny_reparameterization()
