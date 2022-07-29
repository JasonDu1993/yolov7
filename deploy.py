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
from utils.torch_utils import select_device, is_parallel

## YOLOv7 reparameterization
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
                        k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[
                            k].shape}
model.load_state_dict(intersect_state_dict, strict=False)
model.names = ckpt['model'].names
# model.nc = ckpt['model'].nc
model.nc = nc

# reparametrized YOLOR
for i in range(3 * (1 + 5)):
    model.state_dict()['model.105.m.0.weight'].data[i, :, :, :] *= state_dict['model.105.im.0.implicit'].data[:, i,
                                                                   ::].squeeze()
    model.state_dict()['model.105.m.1.weight'].data[i, :, :, :] *= state_dict['model.105.im.1.implicit'].data[:, i,
                                                                   ::].squeeze()
    model.state_dict()['model.105.m.2.weight'].data[i, :, :, :] *= state_dict['model.105.im.2.implicit'].data[:, i,
                                                                   ::].squeeze()
model.state_dict()['model.105.m.0.bias'].data += state_dict['model.105.m.0.weight'].mul(
    state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
model.state_dict()['model.105.m.1.bias'].data += state_dict['model.105.m.1.weight'].mul(
    state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
model.state_dict()['model.105.m.2.bias'].data += state_dict['model.105.m.2.weight'].mul(
    state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
model.state_dict()['model.105.m.0.bias'].data *= state_dict['model.105.im.0.implicit'].data.squeeze()
model.state_dict()['model.105.m.1.bias'].data *= state_dict['model.105.im.1.implicit'].data.squeeze()
model.state_dict()['model.105.m.2.bias'].data *= state_dict['model.105.im.2.implicit'].data.squeeze()

# model to be saved
ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
        'optimizer': None,
        'training_results': None,
        'epoch': -1}

# save reparameterized model
torch.save(ckpt, save_path)
