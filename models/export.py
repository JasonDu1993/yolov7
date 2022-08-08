import argparse
import sys
import time
import numpy as np
import cv2

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from utils.datasets import letterbox


def img_preprocess(image, input_shape):
    # image = Image.open(img_path)
    print("....input")
    img_h, img_w, _ = image.shape
    c, h, w = input_shape  # [c, h, w]
    img = letterbox(image, (h, w), stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = img / 255
    # img = np.stack([img, img])
    img = img[None, ...]
    print("img:", img.flatten()[:20])
    print("img end:", img.flatten()[-20:])
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='/zhoudu/checkpoints/gesture/yolov7/yolov7_416_jsc/weights/deploy.best.pt',
                        help='weights path')
    parser.add_argument('--save_onnx_path', type=str,
                        default='/zhoudu/checkpoints/gesture/yolov7/yolov7_416_jsc/weights/gesture_det.fordpn.onnx',
                        help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[416, 416], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', default=False, help='dynamic ONNX axes')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    # img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection
    img = cv2.imread("imgs/demo.jpg")
    img = img_preprocess(img, (3, 416, 416))
    img = torch.from_numpy(img).to(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    img = img.half() if half else img.float()  # uint8 to fp16/32

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run

    # TorchScript export
    # try:
    #     print('\nStarting TorchScript export with torch %s...' % torch.__version__)
    #     f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
    #     ts = torch.jit.trace(model, img, strict=False)
    #     ts.save(f)
    #     print('TorchScript export success, saved as %s' % f)
    # except Exception as e:
    #     print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        # f = opt.weights.replace('.pt', '.fordpn.onnx')  # filename
        f = opt.save_onnx_path
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['data'],
                          output_names=['classes', 'boxes'] if y is None else ['output0', "output1", "output2"],
                          dynamic_axes={'data': {0: 'batch'},  # size(1,3,640,640)
                                        'output0': {0: 'batch'},
                                        'output1': {0: 'batch'},
                                        'output2': {0: 'batch'},
                                        } if opt.dynamic else None)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # CoreML export
    try:
        import coremltools as ct

        print('\nStarting CoreML export with coremltools %s...' % ct.__version__)
        # convert model from torchscript and apply pixel scaling as per detect.py
        model = ct.convert(ts, inputs=[ct.ImageType(name='image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        model.save(f)
        print('CoreML export success, saved as %s' % f)
    except Exception as e:
        print('CoreML export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
