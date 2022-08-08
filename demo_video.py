# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 17:44
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : demo.py
# @Software: PyCharm
import os
import sys
import argparse
import cv2
import torch
import numpy as np
import onnxruntime
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
# det
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# recog
import torch.nn.functional as F
from gesture_recog.build_model import build_model, get_test_transformer
from utils.draw_box_kpt_utils import draw_box_and_kpt, show_img, save_img
from utils.box_and_kpt_utils import xyxy2xywh, get_area
from utils.class_color_c9 import CLASSES, PALETTE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='yolov7 test (and eval) a model')
    parser.add_argument('--det_ckp',
                        default="weights/det_model/yolov7_tiny_jsc/deploy.best.pt",
                        help='test config file path')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--recog_cfg', default="gesture_recog/exp_c9_affine_hagrid_heart_jsc/config.py",
                        help='test config file path')
    parser.add_argument('--gpu_id', default=-1, type=int, help='test config file path')
    parser.add_argument('--recog_model', default="gesture_recog.exp_c9_affine_hagrid_heart_jsc.model.Model",
                        help='test config file path')
    parser.add_argument('--recog_ckp',
                        default="weights/recog_model/exp_c9_affine_hagrid_heart_jsc/model-epoch19.weights",
                        help='test config file path')
    parser.add_argument('--use_recog_model', default=True, type=bool, help='whether use recog model')
    args = parser.parse_args()

    # 构建检测模型
    gpu_id = int(args.gpu_id)
    device = "cuda:{}".format(gpu_id) if gpu_id >= 0 else "cpu"
    det_model = attempt_load(args.det_ckp, map_location=device)  # load FP32 model
    half = device != 'cpu'  # half precision only supported on CUDA
    if half:
        det_model = det_model.half()
    stride = int(det_model.stride.max())  # model stride
    img_size = args.img_size
    img_size = check_img_size(img_size, s=stride)  # check img_size

    # 构建识别模型
    use_recog_model = args.use_recog_model
    if use_recog_model:
        config = args.recog_cfg
        model = args.recog_model
        weight_path = args.recog_ckp
        gesture_model = build_model(config, model, weight_path)
        gesture_model.to(device)
        trans = get_test_transformer(config)

    cap = cv2.VideoCapture(0)
    cnt = 0
    print("start")
    while cap.isOpened():
        # if cnt % 10 == 0:
        #     time.sleep(1.5)
        ret, frame = cap.read()
        # frame = cv2.imread("t.jpg")
        # show_img(frame)
        image_og_h, image_og_w, _ = frame.shape
        print("image_og_w: {}".format(image_og_w))
        print("image_og_h: {}".format(image_og_h))

        # Padded resize
        img = letterbox(frame, img_size, stride=stride, auto=False)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = det_model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                # Write results
                for j, (x1, y1, x2, y2, score, cls) in enumerate(reversed(det)):
                    xyxy = [x1, y1, x2, y2]
                    bbox = torch.tensor(xyxy).view(1, 4)  # left top w h relative origin image
                    bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
                    bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
                    bbox = bbox.view(-1).tolist()
                    print("pred box {} sco {}".format(bbox, score))
                    if use_recog_model:
                        kpts = None
                        img_path = None
                        img_new, bbox_new, kpts, img_path = trans(frame, bbox, kpts, img_path)
                        img_new = img_new.unsqueeze(0)
                        if gpu_id >= 0:
                            img_new = img_new.to(gpu_id)

                        ret = gesture_model(img_new, mode="test")
                        if isinstance(ret, tuple):
                            ret = ret[0]
                        # 使用proser方法的模型进行测试
                        # ret = F.softmax(ret, dim=1)
                        # ret = ret.cpu().detach().numpy()
                        # dummyconf = ret[:, -1]
                        # maxknownconf = np.max(ret[:, :-1], axis=1)
                        # conf = maxknownconf - dummyconf

                        # 使用原始的方法进行训练的模型进行测试
                        ret = F.softmax(ret, dim=1)
                        ret = ret.cpu().detach().numpy()
                        conf = np.max(ret, axis=1)

                        pred_label = ret.argmax(axis=1)[0]
                        pred_score = ret.max(axis=1)[0]
                        # if conf < 0.6208:
                        #     pred_label = len(CLASSES)
                        # img = cv2.imread(img_path)
                        cls_name = "neg" if pred_label == len(CLASSES) or pred_label == -1 else CLASSES[pred_label]
                        box_txt = "{}:{:.2f}".format(cls_name, pred_score)
                        print("recog: {}".format(box_txt))
                        img = draw_box_and_kpt(frame, bbox, box_color=PALETTE[pred_label], box_txt=box_txt)
                        # show_img(img)
                # img = cv2.resize(img, (100, 100))
                cv2.imshow('Frame', img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
