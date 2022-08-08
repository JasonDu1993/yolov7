# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 18:07
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : get_box_label.py
# @Software: PyCharm
import os
import json
import cv2
from utils.draw_box_kpt_utils import draw_box_and_kpt, show_img, save_img
import matplotlib.pyplot as plt
from utils.get_path_len import get_path_len


def get_common_scenario_mobile_box_label():
    src_paths = [
        # "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_cloudwalk.txt",
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_imgs_1015_1017.txt",
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_imgs_1018_1019.txt",
    ]
    dst_paths = [
        # "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_cloudwalk.box.txt",
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_imgs_1015_1017.box.txt",
        "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_imgs_1018_1019.box.txt",
    ]
    for src_path, dst_path in zip(src_paths, dst_paths):
        print("src: {}".format(src_path))
        print("dst: {}".format(dst_path))
        with open(src_path, "r", encoding="utf-8") as fr:
            with open(dst_path, "w", encoding="utf-8") as fw:
                for line in fr.readlines():
                    line_sp = line.strip().split(" ")
                    img_path = line_sp[0]
                    # img = cv2.imread(img_path)
                    label = line_sp[1]
                    width, height = int(line_sp[2]), int(line_sp[3])
                    json_path = img_path + ".json"
                    # print("json_path: {}".format(json_path))
                    if not os.path.exists(json_path):
                        print("  json not exist: {}".format(json_path))
                        continue
                    print(json_path)
                    try:
                        fj = open(json_path, "r", encoding="utf-8")
                        labels = json.load(fj)
                    except UnicodeDecodeError:
                        print("use utf-8 encoding read file error")
                        fj = open(json_path, "r", encoding="gbk")
                        labels = json.load(fj)
                    label_ai = labels[0]
                    x1 = 10000
                    x2 = 0
                    y1 = 10000
                    y2 = 0
                    if "coordinates" not in label_ai:
                        print(label_ai)
                        continue
                    if "负例" == label_ai["name"]:
                        continue
                    for kpt in label_ai["coordinates"]:
                        # print(kpt)
                        x = float(kpt["axisX"])
                        y = float(kpt["axisY"])
                        # index = int(kpt["index"])
                        if x < x1:
                            x1 = x
                        if x > x2:
                            x2 = x
                        if y < y1:
                            y1 = y
                        if y > y2:
                            y2 = y
                    box = [x1, y1, x2 - x1, y2 - y1]  # x, y, w, h
                    # print(box)
                    new_line = img_path + " " + str(label) + " " + str(width) + " " + str(height) + " " + " ".join(
                        list(map(lambda x: "%.2f" % x, box))) + "\n"
                    # print(new_line)
                    fw.write(new_line)
                    fw.flush()
                    # break


def get_indoor_multi_scenario_mobile_box_label():
    import xml.etree.ElementTree as ET
    src_paths = [
        "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/txt/indoor_multi_scenario.txt",
    ]
    dst_paths = [
        "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/txt/indoor_multi_scenario.box.1.txt",
    ]
    vis_dst_paths = [
        "/zhoudu/golabel/demo/cluster_rst/indoor_multi_scenario.box.vis.1.txt",
    ]
    cls_desc_paths = [
        "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/txt/class_description.1.txt",
    ]
    data_root = "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/data"
    label_root = "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/labels_xml"
    save_root = "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/vis/imgs"
    vis = False
    if vis:
        os.makedirs(save_root, exist_ok=True)
    for i, src_path in enumerate(src_paths):
        dst_path = dst_paths[i]
        cls_desc_path = cls_desc_paths[i]
        vis_dst_path = vis_dst_paths[i]
        print("src: {}".format(src_path))
        print("dst: {}".format(dst_path))
        print("cls_desc_path: {}".format(cls_desc_path))
        print("vis_dst_path: {}".format(vis_dst_path))
        clsname_to_idx = {}
        cnt = -1
        with open(src_path, "r", encoding="utf-8") as fr:
            with open(dst_path, "w", encoding="utf-8") as fw:
                with open(vis_dst_path, "w", encoding="utf-8") as fw1:
                    for line in fr.readlines():
                        line_sp = line.strip().split(" ")
                        img_path = line_sp[0]
                        label = line_sp[1]
                        width, height = int(line_sp[2]), int(line_sp[3])
                        img_path_sp = img_path.split(".")
                        img_path_sp[-1] = "xml"
                        label_img_path = ".".join(img_path_sp)
                        xml_path = os.path.join(label_root, label_img_path[get_path_len(data_root):])
                        print("xml_path: {}".format(xml_path))
                        if not os.path.exists(xml_path.encode("utf-8")):
                            print("label path not exist: {}".format(xml_path))
                            continue
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        cls_name = root.find('object/name').text  # 只返回找到的第一个
                        if cls_name not in clsname_to_idx:
                            cnt += 1
                            clsname_to_idx[cls_name] = cnt
                        xmin = float(root.find('object/bndbox/xmin').text)  # 只返回找到的第一个
                        ymin = float(root.find('object/bndbox/ymin').text)  # 只返回找到的第一个
                        xmax = float(root.find('object/bndbox/xmax').text)  # 只返回找到的第一个
                        ymax = float(root.find('object/bndbox/ymax').text)  # 只返回找到的第一个
                        if "5s-" in img_path or "iphone6-" in img_path or "iphone7-" in img_path:
                            o_h = width
                            o_w = height
                            new_xmin = o_h - ymax
                            new_ymin = xmin
                            new_w = ymax - ymin
                            new_h = xmax - xmin
                            box = [new_xmin, new_ymin, new_w, new_h]
                        else:
                            box = [xmin, ymin, xmax - xmin, ymax - ymin]

                        new_line = img_path + " " + str(clsname_to_idx[cls_name]) + " " + str(width) + " " + str(
                            height) + " " + " ".join(list(map(lambda x: "%.2f" % x, box))) + "\n"
                        # print(new_line)
                        fw.write(new_line)
                        fw.flush()
                        if vis:
                            img = cv2.imread(img_path)
                            img = draw_box_and_kpt(img, box, box_color=(255, 0, 0),
                                                   box_txt="gt:{}".format(clsname_to_idx[cls_name]))
                            save_img_path = os.path.join(save_root, img_path[get_path_len(data_root):])
                            save_img(save_img_path, img)
                            vis_line = save_img_path + " " + str(clsname_to_idx[cls_name]) + " 1\n"
                            fw1.write(vis_line)
                            fw1.flush()

        with open(cls_desc_path, "w", encoding="utf-8") as fw2:
            for name in clsname_to_idx:
                idx = clsname_to_idx[name]
                new_line = str(idx) + " " + name + "\n"
                fw2.write(new_line)
                fw2.flush()


def get_indoor_office_mobile_box_label():
    import xml.etree.ElementTree as ET
    src_paths = [
        "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/txt/indoor_office.txt",
    ]
    dst_paths = [
        "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/txt/indoor_office.box.txt",
    ]
    vis_dst_paths = [
        "/zhoudu/golabel/demo/cluster_rst/indoor_office.box.vis.txt",
    ]
    cls_desc_paths = [
        "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/txt/class_description.txt",
    ]
    data_root = "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/data"
    label_root = "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/labels_xml"
    save_root = "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/vis/imgs"
    vis = False
    if vis:
        os.makedirs(save_root, exist_ok=True)
    for i, src_path in enumerate(src_paths):
        dst_path = dst_paths[i]
        cls_desc_path = cls_desc_paths[i]
        vis_dst_path = vis_dst_paths[i]
        print("src: {}".format(src_path))
        print("dst: {}".format(dst_path))
        print("cls_desc_path: {}".format(cls_desc_path))
        print("vis_dst_path: {}".format(vis_dst_path))
        clsname_to_idx = {}
        cnt = -1
        with open(src_path, "r", encoding="utf-8") as fr:
            with open(dst_path, "w", encoding="utf-8") as fw:
                with open(vis_dst_path, "w", encoding="utf-8") as fw1:
                    for line in fr.readlines():
                        line_sp = line.strip().split(" ")
                        img_path = line_sp[0]
                        # if "iphone6" not in img_path:
                        #     continue
                        label = line_sp[1]
                        width, height = int(line_sp[2]), int(line_sp[3])
                        img_path_sp = img_path.split(".")
                        img_path_sp[-1] = "xml"
                        label_img_path = ".".join(img_path_sp)
                        xml_path = os.path.join(label_root, label_img_path[get_path_len(data_root):])
                        print("xml_path: {}".format(xml_path))
                        if not os.path.exists(xml_path.encode("utf-8")):
                            print("label path not exist: {}".format(xml_path))
                            continue
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        cls_name = root.find('object/name').text  # 只返回找到的第一个
                        if cls_name not in clsname_to_idx:
                            cnt += 1
                            clsname_to_idx[cls_name] = cnt
                        xmin = float(root.find('object/bndbox/xmin').text)  # 只返回找到的第一个
                        ymin = float(root.find('object/bndbox/ymin').text)  # 只返回找到的第一个
                        xmax = float(root.find('object/bndbox/xmax').text)  # 只返回找到的第一个
                        ymax = float(root.find('object/bndbox/ymax').text)  # 只返回找到的第一个

                        if "iphone6" in img_path:
                            o_h = width
                            o_w = height
                            new_xmin = o_h - ymax
                            new_ymin = xmin
                            new_w = ymax - ymin
                            new_h = xmax - xmin
                            box = [new_xmin, new_ymin, new_w, new_h]
                        else:
                            box = [xmin, ymin, xmax - xmin, ymax - ymin]

                        new_line = img_path + " " + str(clsname_to_idx[cls_name]) + " " + str(width) + " " + str(
                            height) + " " + " ".join(list(map(lambda x: "%.2f" % x, box))) + "\n"
                        # print(new_line)
                        fw.write(new_line)
                        fw.flush()
                        if vis:
                            img = cv2.imread(img_path)
                            img = draw_box_and_kpt(img, box, box_color=(255, 0, 0),
                                                   box_txt="gt:{}".format(clsname_to_idx[cls_name]))
                            save_img_path = os.path.join(save_root, img_path[get_path_len(data_root):])
                            save_img(save_img_path, img)
                            vis_line = save_img_path + " " + str(clsname_to_idx[cls_name]) + " 1\n"
                            fw1.write(vis_line)
                            fw1.flush()

        with open(cls_desc_path, "w", encoding="utf-8") as fw2:
            for name in clsname_to_idx:
                idx = clsname_to_idx[name]
                new_line = str(idx) + " " + name + "\n"
                fw2.write(new_line)
                fw2.flush()


def gen_hagrid_subsample():
    # src_paths = os.listdir("/dataset/dataset/ssd/gesture/hagrid/subsample/ann_subsample")
    src_paths = ["/dataset/dataset/ssd/gesture/hagrid/subsample/ann_subsample/mute.json"]
    img_root = "/dataset/dataset/ssd/gesture/hagrid/subsample/imgs"
    cls_desc_path = "/dataset/dataset/ssd/gesture/hagrid/subsample/labels/class_description.mute.txt"
    save_path = "/dataset/dataset/ssd/gesture/hagrid/subsample/labels/subsample.mute.txt"
    # clsname_to_idx = {"no_gesture": -1}
    clsname_to_idx = {}
    cnt = -1
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fw:
        for src_path in src_paths:
            cls = src_path.split("/")[-1].split(".")[0]
            f = open(src_path, "r", encoding="utf-8")
            anns = json.load(f)
            print(anns.keys())

            for name in anns.keys():
                ann = anns[name]
                img_path = os.path.join(img_root, cls, name + ".jpg")
                img = cv2.imread(img_path)
                imgh, imgw, imgc = img.shape
                boxes = ann["bboxes"]
                labels = ann["labels"]
                for i, box in enumerate(boxes):
                    cls_name = labels[i]
                    if cls_name == "no_gesture":
                        continue
                    if cls_name not in clsname_to_idx:
                        cnt += 1
                        clsname_to_idx[cls_name] = cnt
                    x, y, w, h = box
                    x = x * imgw
                    y = y * imgh
                    w = w * imgw
                    h = h * imgh
                    box = [x, y, w, h]
                    new_line = img_path + " " + str(clsname_to_idx[cls_name]) + " " + str(imgw) + " " + str(
                        imgh) + " " + " ".join(list(map(lambda x: "%.2f" % x, box))) + "\n"
                    # print(new_line)
                    fw.write(new_line)
                    fw.flush()
                    # img = cv2.imread(img_path)
                    # img = draw_box_and_kpt(img, box, box_color=(255, 0, 0),
                    #                        box_txt="gt:{}".format(clsname_to_idx[cls_name]))
                    # show_img(img)
    with open(cls_desc_path, "w", encoding="utf-8") as fw2:
        for name in clsname_to_idx:
            idx = clsname_to_idx[name]
            new_line = str(idx) + " " + name + "\n"
            fw2.write(new_line)
            fw2.flush()


def gen_hagrid():
    ann_root = "/dataset/dataset/ssd/gesture/hagrid/train_val/ann_train_val"
    src_paths = [os.path.join(ann_root, name) for name in sorted(os.listdir(ann_root))]
    # src_paths = ["/dataset/dataset/ssd/gesture/hagrid/subsample/ann_subsample/mute.json"]
    img_root = "/dataset/dataset/ssd/gesture/hagrid/train_val/imgs"
    cls_desc_path = "/dataset/dataset/ssd/gesture/hagrid/txt/class_description.txt"
    save_path = "/dataset/dataset/ssd/gesture/hagrid/txt/train.hagrid.txt"
    clsname_to_idx = {"no_gesture": -1}
    # clsname_to_idx = {}
    cnt = -1
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fw:
        for src_path in src_paths:
            print("src_path:{}".format(src_path))
            cls = src_path.strip().split("/")[-1].split(".")[0]
            if cls not in ["one", "peace", "peace_inverted", "three", "four", "palm", "fist", "ok", "like"]:
                continue
            print("src_path:{}".format(src_path))
            print("cls name :{}".format(cls))
            f = open(src_path, "r", encoding="utf-8")
            anns = json.load(f)

            for name in anns.keys():
                ann = anns[name]
                img_path = os.path.join(img_root, "train_val_" + cls, name + ".jpg")
                img = cv2.imread(img_path)
                imgh, imgw, imgc = img.shape
                boxes = ann["bboxes"]
                labels = ann["labels"]
                for i, box in enumerate(boxes):
                    cls_name = labels[i]
                    if cls_name == "no_gesture":
                        continue
                    if cls_name not in clsname_to_idx:
                        cnt += 1
                        clsname_to_idx[cls_name] = cnt
                    x, y, w, h = box
                    x = x * imgw
                    y = y * imgh
                    w = w * imgw
                    h = h * imgh
                    box = [x, y, w, h]
                    new_line = img_path + " " + str(clsname_to_idx[cls_name]) + " " + str(imgw) + " " + str(
                        imgh) + " " + " ".join(list(map(lambda x: "%.2f" % x, box))) + "\n"
                    # print(new_line)
                    fw.write(new_line)
                    fw.flush()
                    # img = cv2.imread(img_path)
                    # img = draw_box_and_kpt(img, box, box_color=(255, 0, 0),
                    #                        box_txt="gt:{}".format(clsname_to_idx[cls_name]))
                    # show_img(img)
    with open(cls_desc_path, "w", encoding="utf-8") as fw2:
        for name in clsname_to_idx:
            idx = clsname_to_idx[name]
            new_line = str(idx) + " " + name + "\n"
            fw2.write(new_line)
            fw2.flush()


if __name__ == '__main__':
    # get_common_scenario_mobile_box_label()
    # get_indoor_multi_scenario_mobile_box_label()
    # get_indoor_office_mobile_box_label()
    # gen_hagrid_subsample()
    gen_hagrid()
