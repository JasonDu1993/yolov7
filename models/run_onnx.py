import argparse
import os
import sys
import time
import numpy as np
import cv2
import onnxruntime

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression


def img_preprocess(image, input_shape):
    # image = Image.open(img_path)
    print("....input")
    img_h, img_w, _ = image.shape
    c, h, w = input_shape  # [c, h, w]
    img = letterbox(image, (h, w), stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img).astype(np.float32)
    img = img / 255
    # img = np.stack([img, img])
    img = img[None, ...]
    print("img:", img.flatten()[:20])
    print("img end:", img.flatten()[-20:])
    print("img at 30000:", img.flatten()[30000:30000+20])
    return img


def run_onnx(onnx_path, input_shape):
    # sess_options = onnxruntime.SessionOptions()
    # print(sess_options.graph_optimization_level)

    # sess_options.intra_op_num_threads = 1

    # Set graph optimization level
    # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # To enable model serialization after graph optimization set this
    # sess_options.optimized_model_filepath = "./model/optimized_synboost_dynamic_batch_hw.onnx"
    sess_options = None
    model = onnxruntime.InferenceSession(onnx_path, sess_options=sess_options, providers=['CUDAExecutionProvider'])
    print("get_providers", model.get_providers())
    print("get_provider_options", model.get_provider_options())
    print("graph_optimization_level", model.get_session_options().graph_optimization_level)
    root = "./imgs"
    names = list(os.listdir(root))
    for i, name in enumerate(names):
        img_path = os.path.join(root, name)
        print("img_path: {}".format(img_path))
        # img_path = "./sample_images/road_anomaly_example.png"
        # img_path = "/zhoudu/workspaces/road_obstacles/sample_images/onnx_conversion/leftImg8bit/val/frankfurt/frankfurt_000001_083852_leftImg8bit.png"
        # mask_path = "/zhoudu/workspaces/road_obstacles/sample_images/onnx_conversion/gtFine/val/frankfurt/frankfurt_000001_083852_gtFine_labelIds.png"
        image = cv2.imread(img_path)
        image_og_h, image_og_w, _ = image.shape
        x = img_preprocess(image, input_shape)
        input_name = model.get_inputs()[0].name
        print("input_name:", input_name, x.shape)
        # for i in range(100):
        # torch.cuda.synchronize()
        t0 = time.time()
        # print("i:", i)
        outputs = model.run(None, {input_name: x})
        for out in outputs:
            print(out.shape, out.flatten()[:20])
        # torch.cuda.synchronize()
        t1 = time.time()
        print("onnx run {} s".format(t1 - t0))
        # Apply NMS
        conf_thres = 0.1
        iou_thres = 0.45
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None,
                                   agnostic=False)



        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s += '%gx%g ' % (img_h, img_w)  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str,
                        default='/zhoudu/checkpoints/gesture/yolov7/yolov7_tiny_irlight/weights/gesture_det.fordpn.v1.0.0.20220825.onnx',
                        help='weights path')
    parser.add_argument('--input_shape', nargs='+', type=int, default=[3, 416, 416], help='image size')  # c, height, width
    args = parser.parse_args()
    print(args)
    run_onnx(args.onnx_path, args.input_shape)

