# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 14:35
# @Author  : 暖枫
# @Email   : zhoudu@cloudwalk.com
# @File    : train.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.autograd.profiler
import torch.nn.functional as F
import sys, os, copy, time
import numpy as np
from torch.nn import BatchNorm1d
import torchvision
import re
from collections import OrderedDict
from torchvision.models.resnet import ResNet, BasicBlock, model_urls
from torch.hub import load_state_dict_from_url
from gesture_recog.shufflenetv2 import ShuffleNetV2_Plus
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

print("class path:", __file__)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.parallel_type = 'ddp'
        self.shape = args

    def forward(self, x):
        return x.view(x.size(0), *self.shape).contiguous()


class Model(nn.Module):
    def __init__(self, input_shape, class_num=14, garbage_class_num=10, deploy=False):
        """

        Args:
            input_shape: [C, H, W]
            emb_size: 5 * 2，5 keypoint, left_eye, right_eye, nose, left_mouse, right_mouse, 2 represent x, y
        """
        super(Model, self).__init__()
        self.parallel_type = 'ddp'
        self.input_shape = input_shape
        self.class_num = class_num
        self.garbage_class_num = garbage_class_num
        self.deploy = deploy
        num_features = 128
        # self.backbone = iresnet18(num_features=num_features)
        # 使用预训练模型
        model = ShuffleNetV2_Plus(input_shape[1])
        # 提取fc层中固定的参数
        # fc_features = model.classifier.in_features
        # 修改类别为10，重定义最后一层
        # model.fc = nn.Linear(fc_features, num_features)
        self.backbone = model
        self.fc = nn.Linear(1280, self.class_num + self.garbage_class_num, bias=False)
        # init.init_cnn(self.fc.modules(), force=False)
        if os.path.exists("/dataset/dataset/ssd/model_meta/ShuffleNetV2+/ShuffleNetV2+.Large.pth.tar"):
            load_weight(self.backbone, "/dataset/dataset/ssd/model_meta/ShuffleNetV2+/ShuffleNetV2+.Large.pth.tar", strict=False)

    def forward(self, x, y=None, alpha=0, mode="test"):
        x = self.backbone(x)
        x = self.fc(x)
        if alpha > 0:
            mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha)
            return x, mixed_x, y_a, y_b, lam
        if mode == "onnx":
            x = F.softmax(x, dim=1)
            sco, label = x.max(dim=1)
            label = label.float()
            return label, sco
        return x, x


def load_weight(net, weights, ignore_patten=None, deploy=False, strict=True):
    print("load weight {}".format(weights))
    checkpoint = torch.load(weights, map_location='cpu')
    if "weights" in checkpoint.keys():
        checkpoint = checkpoint['weights']
    if "net_state_dict" in checkpoint.keys():
        checkpoint = checkpoint['net_state_dict']
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    if deploy:
        print("Convert acblock weight to deploy mode...")
        checkpoint = convert_acnet_weights(checkpoint, eps=1e-5)
    if ignore_patten is not None:
        strict = False
        new_net_states = OrderedDict()
        for k, w in checkpoint.items():
            if re.search(ignore_patten, k) is None:
                new_net_states[k] = w
            else:
                print("ignore {} shape {}".format(k, w.shape))
        checkpoint = new_net_states
    net.load_state_dict(checkpoint, strict=strict)


if __name__ == '__main__':
    input_shape = (3, 64, 64)
    kpt_num = 5
    device = "cuda:1"
    deploy = True
    model = Model(input_shape=input_shape, kpt_num=kpt_num)
    model.eval()
    model.to(device)
    for name in model.state_dict():
        print(name)
    # weights = "/zhoudu/checkpoints/r3an/fb202102/r3an20210207b_101056/checkpoint/r3an-epoch29.weights"
    # checkpoint = torch.load(weights, map_location='cpu')['weights']
    # import pprint
    #
    # pprint.pprint(list(checkpoint.keys()))
    print("<<<<<")
    weights = "/zhoudu/checkpoints/kpt_models/kpt211207a_193755/checkpoint/kpt-epoch199.weights"
    load_weight(model, weights, deploy=deploy)

    inputs = torch.ones((2, 3, 112, 96), dtype=torch.float32, device=device)
    print("inputs:", inputs.shape, inputs[0, :3, :3, :3])
    outputs = model(inputs)
    print("output:", outputs.shape, outputs[:, :10])

    # # Warm-up
    # for _ in range(5):
    #     start = time.time()
    #     outputs = model(inputs)
    #     torch.cuda.synchronize(torch.device(device))
    #     end = time.time()
    #     print('Time:{}ms'.format((end - start) * 1000))
    #
    # # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True) as prof:
    # #     outputs = model(inputs)
    # # print(prof.table())
    # # prof.export_chrome_trace('./r3an_profile.json')
    #
    # import torchprof
    #
    # # method 1  https://github.com/awwong1/torchprof
    # with torchprof.Profile(model, use_cuda=True) as prof:
    #     model(inputs)
    # print(prof.display(show_events=False))  # equivalent to `print(prof)` and `print(prof.display())`

    # # method 2
    # with torchprof.Profile(model, use_cuda=True) as prof:
    #     model(inputs)
    # trace, event_lists_dict = prof.raw()
    # # print("trace", trace[1])
    # # Trace(path=('AlexNet', 'features', '0'), leaf=True, module=Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)))
    # # print(event_lists_dict[trace[1].path][0])
    # paths = [("FaceBridge", "deconv1", "0"),
    #          ("FaceBridge", "deconv2", "0"),
    #          ]
    # for path in paths:
    #     print(path)
    #     print(event_lists_dict[path][0])
    #
    # # method 3
    # # Layer does not have to be a leaf layer
    # paths = [("FaceBridge", "deconv1", "0"), ("FaceBridge", "deconv2", "0")]
    # with torchprof.Profile(model, use_cuda=True, paths=paths) as prof:
    #     model(inputs)
    # print(prof)
