# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 10:53
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : gen_txt.py
# @Software: PyCharm
import os
from natsort import natsorted
from utils.get_path_len import get_path_len


def traverse(src_dir, exts=None):
    if exts is None:
        exts = ['.jpg', '.png', '.bmp']
    labels = {}
    result = []
    cnt = -1
    for root, dirs, files in natsorted(os.walk(src_dir)):
        if len(files) > 0:
            class_name = root[get_path_len(src_dir):].split("/")[0]
            if class_name not in labels:
                cnt += 1
                labels[class_name] = str(cnt)
            for f in natsorted(files):
                ext = os.path.splitext(f)[-1].lower()
                if ext in exts:
                    result.append([os.path.join(root, f), str(cnt)])
    print(labels)
    return result, labels


def gen_txt(src_dir, save_path, save_label_path=None):
    result, labels = traverse(src_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("save_path: {}".format(save_path))
    with open(save_path, "w", encoding="utf-8") as fw:
        for lines in result:
            img_path, class_name = lines
            color = "1"
            new_line = img_path + " " + class_name + " " + color + "\n"
            fw.write(new_line)
    if save_label_path is not None:
        print("save_label_path: {}".format(save_label_path))
        os.makedirs(os.path.dirname(save_label_path), exist_ok=True)
        with open(save_label_path, "w", encoding="utf-8") as fw:
            for key, value in labels.items():
                line = value + " " + key + "\n"
                fw.write(line)


if __name__ == '__main__':
    src_dir = "/zhoudu/test/gesture/show1/yolo_for_gesture/yolov3_d53_fp16_mstrain-416_273e_coco/epoch_273.pth/common_scenario_mobile_201021/ges30_imgs_1015_1017"
    save_path = "/zhoudu/golabel/demo/cluster_rst/yolov3_ges30_imgs_1015_1017.txt"
    # save_label_path = "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txt/class_description.ges30_imgs_1015_1017.txt"
    save_label_path = None
    gen_txt(src_dir, save_path, save_label_path)
