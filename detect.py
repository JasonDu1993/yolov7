import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import matplotlib.pyplot as plt
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_txt_path = opt.save_txt_path
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    print("view_img:{}".format(view_img))
    print("save_img:{}".format(save_img))
    webcam = source.isnumeric() or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    random.seed(1234)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
    input_img_num = 0
    with open(save_txt_path, "w", encoding="utf-8") as fw:
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            input_img_num += img.shape[0]
            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                # p = Path(p)  # to Path
                img_h, img_w = im0.shape[:2]
                s += '%gx%g ' % (img_h, img_w)  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += "{} {}{}, ".format(n, names[int(c)], 's' * (n > 1))  # add to string

                    # Write results
                    for x1, y1, x2, y2, conf, cls in reversed(det):
                        xyxy = [x1, y1, x2, y2]
                        if save_txt:  # Write to file
                            # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            lxlywh = torch.tensor(xyxy).view(1, 4)  # left top w h relative origin image
                            lxlywh[:, 2] = lxlywh[:, 2] - lxlywh[:, 0]
                            lxlywh[:, 3] = lxlywh[:, 3] - lxlywh[:, 1]
                            lxlywh = lxlywh.view(-1).tolist()
                            # line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            new_line = path + " " + str(int(cls))
                            if opt.contain_wh:
                                new_line += " " + str(img_w) + " " + str(img_h)
                            new_line += " " + " ".join(list(map(lambda x: "{:.2f}".format(x), lxlywh)))  # save box
                            if opt.save_conf:
                                new_line += " " + "{:.2f}".format(float(conf))
                            new_line += "\n"
                            # print("new_line:{}".format(new_line))
                            fw.write(new_line)
                            fw.flush()

                        if save_img or view_img:  # Add bbox to image
                            label = '{} {:.2f}'.format(names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                # print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if view_img:
                    # cv2.imshow(str(p), im0)
                    # cv2.waitKey(1)  # 1 millisecond
                    plt.imshow(im0[:, :, ::-1])
                    plt.show()

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(" The image with the result is saved in: {}".format(save_path))
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

    print('Done. ({:.3f}s, {} images)'.format(time.time() - t0, input_img_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/zhoudu/checkpoints/gesture/yolov7/yolov7_tiny_gray/weights/deploy.best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='/dataset/dataset/ssd/gesture/leapGestRecog/c9/leapGestRecog.map.txt',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False, help='display results')
    parser.add_argument('--save-txt', default=True, help='save results to *.txt')
    parser.add_argument('--save-txt-path',
                        default="/dataset/dataset/ssd/gesture/leapGestRecog/c9/det.yolov7_tiny_gray.leapGestRecog.map.txt",
                        help='save results to one file')
    parser.add_argument('--contain-wh', default=True, help='save the image width and height in --save-txt labels')
    parser.add_argument('--save-conf', default=True, help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', default=True, help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', default=True, help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))
    print("src_path: {}".format(opt.source))
    print("save_path: {}".format(opt.save_txt_path))
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
