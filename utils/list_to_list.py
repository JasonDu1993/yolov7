# -*- coding: utf-8 -*-
# @Time    : 2021/9/30 17:39
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : align_list_to_orig_list.py
# @Software: PyCharm
import os, sys
import shutil
import json
import cv2
import numpy as np
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.box_and_kpt_utils import IOU_XYWH
from utils.box_and_kpt_utils import get_enclose_bbox_of_kpt


def get_unique_name(img_path, root=None, isaligin=False):
    if root is not None:
        img_path = img_path[len(root):]
    img_name = os.path.splitext(img_path)[0]
    if isaligin:  # 如果是对齐图，则将对齐图自动生成的hash后缀去掉
        img_name = "_".join(img_name.split("_")[:-1])
    return img_name


def align_list_to_orign_list(low_qual_path, save_low_qual_orign_path, origin_img_dir):
    with open(low_qual_path, "r") as fr:
        with open(save_low_qual_orign_path, "w") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                img_name = os.path.basename(img_path)
                img_name_sp = os.path.splitext(img_name)
                origin_img_name = "_".join(img_name_sp[0].split("_")[:-1]) + img_name_sp[1]
                # print(origin_img_name)
                img_path = os.path.join(origin_img_dir, origin_img_name)
                fw.write(img_path + " " + line_sp[1] + " 1 " + line_sp[3] + "\n")


def orign_list_to_det_list(origin_list, det_list_src, det_list_dst):
    sources = {}
    with open(det_list_src, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            name = get_unique_name(line_sp[0])
            sources[name] = line
    with open(origin_list, "r") as fr:
        with open(det_list_dst, "w") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                name = get_unique_name(line_sp[0])
                new_line = sources[name]
                fw.write(new_line)


def gen_shenyang500(src_path, det_label_path, save_tang_label_path):
    sources = {}
    with open(det_label_path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            name = get_unique_name(line_sp[0])
            sources[name] = line
    with open(src_path, "r") as fr:
        with open(save_tang_label_path, "w") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                name = get_unique_name(line_sp[0])
                src_line = sources[name]
                src_line_sp = src_line.strip().split(" ")
                img_path = src_line_sp[0]
                img = cv2.imread(img_path)
                height, width, c = img.shape
                x, y, w, h = list(map(float, src_line_sp[2:6]))
                x2 = x + w
                y2 = y + h
                box = [str(y), str(x), str(y2), str(x2)]
                box_str = " ".join(box)
                kpt = []
                for i in range(5):
                    kpt.append(src_line_sp[6 + i * 2 + 1])
                    kpt.append(src_line_sp[6 + i * 2])
                kpt_str = " ".join(kpt)
                new_line = img_path + " " + str(height) + " " + str(width) + " " + box_str + " " + kpt_str + "\n"
                fw.write(new_line)


def tang_list_to_draw_list(src_tang_list, src_draw_lis, dst_draw_list):
    sources = {}
    with open(src_draw_lis, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            name = get_unique_name(line_sp[0])
            sources[name] = line
    with open(src_tang_list, "r") as fr:
        with open(dst_draw_list, "w") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                name = get_unique_name(line_sp[0])
                src_line = sources[name]
                fw.write(src_line)


def gen_list_from_dir(root_dir, save_path):
    cnt = 0
    label = 0
    with open(save_path, "w") as f:
        for root, dirs, files in os.walk(root_dir):
            for fil in files:
                f_path = os.path.join(root, fil)
                if cnt % 20 == 0:
                    label += 1
                cnt += 1
                s = f_path + " " + str(label) + " 1\n"
                f.write(s)
    print(cnt)


def filter_by_fiq():
    path = "/dataset/dataset/ssd/kpt/cluster_rst/sy.ssd.filter.draw.list"
    save_path = "/dataset/dataset/ssd/kpt/cluster_rst/shenyang_test200.draw.list"
    score_thd = 0.00
    num = 60000

    cnt = 0
    with open(path, "r") as fr:
        with open(save_path, "w") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                score = float(line_sp[-1])
                if score <= score_thd:
                    cnt += 1
                    if cnt >= num + 1 and cnt <= num + 200:
                        fw.write(line)
                    elif cnt <= num:
                        continue
                    else:
                        break


def cwface_label_to_tang_label(src_path, dst_path):
    with open(src_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                img = cv2.imread(img_path)
                height, width, c = img.shape
                x, y, w, h = list(map(float, line_sp[2:6]))
                x2 = x + w
                y2 = y + h
                box = [str(y), str(x), str(y2), str(x2)]
                box_str = " ".join(box)
                kpt = []
                for i in range(5):
                    kpt.append(line_sp[6 + i * 2 + 1])
                    kpt.append(line_sp[6 + i * 2])
                kpt_str = " ".join(kpt)
                new_line = img_path + " " + str(height) + " " + str(width) + " " + box_str + " " + kpt_str + "\n"
                fw.write(new_line)


def tang_label_to_cwface_label(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                # img = cv2.imread(img_path)
                # height, width, c = img.shape
                img_h, img_w = int(line_sp[1]), int(line_sp[2])
                y0, x0, y1, x1 = list(map(float, line_sp[3:7]))
                w = x1 - x0
                h = y1 - y0
                box = [str(x0), str(y0), str(w), str(h)]
                box_str = " ".join(box)
                kpt = []
                for i in range(5):
                    kpt.append(line_sp[7 + i * 2 + 1])
                    kpt.append(line_sp[7 + i * 2])
                kpt_str = " ".join(kpt)
                new_line = img_path + " -1 " + box_str + " " + kpt_str + "\n"
                fw.write(new_line)


def select_some_sample(src_path, dst_path, num=4000, base=4):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cnt = 0
    with open(src_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for index, line in enumerate(fr.readlines()):
                if cnt >= num:
                    break
                if index % base < 2:
                    cnt += 1
                    fw.write(line)


def delete_some_col(src_path, dst_path):
    with open(src_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                new_line = [line_sp[0], "-1"]
                new_line.extend(line_sp[3:])
                new_line_str = " ".join(new_line) + "\n"
                fw.write(new_line_str)


def match_max_iou(box, lines, use_kpt_box=False, thd=0.1, is_contain_imgwh=True):
    assert isinstance(lines, list)
    max_iou = -1
    if is_contain_imgwh:
        src_box_pos = 4
    else:
        src_box_pos = 2
    match_line = lines[0]
    for line in lines:
        line_sp = line.strip().split(" ")
        if use_kpt_box:
            kpt = list(map(float, line_sp[src_box_pos + 4:src_box_pos + 4 + 10]))  # x, y
            box1 = get_enclose_bbox_of_kpt(kpt)
        else:
            box1 = list(map(float, line_sp[src_box_pos:src_box_pos + 4]))  # x, y, w, h
        iou = IOU_XYWH(box, box1)
        if max_iou < iou:
            match_line = line
            max_iou = iou
    if max_iou >= thd:
        return match_line
    else:
        return None


def modify_seq(src_path, base_path, dst_path, is_use_box=True, is_contain_imgwh=True, thd=0.5, verbose=False):
    """dst_path的显示顺序和base_path的一致,，内容来自于src_path, 主要用于找到src_path中某张图片的框和base_path对应图片框相匹配的内容进行存储

    Args:
        src_path:
        base_path:
        dst_path:
    """
    print("modify sequence, is_use_box: {}".format(is_use_box))
    source = OrderedDict()
    with open(src_path, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            if img_path not in source:
                source[img_path] = [line]
            else:
                source[img_path].append(line)
    skip_imgs = []

    if is_use_box:
        if is_contain_imgwh:
            src_box_pos = 4
        else:
            src_box_pos = 2
        with open(base_path, "r", encoding="utf-8") as fr:
            with open(dst_path, "w", encoding="utf-8") as fw:
                for index, line in enumerate(fr.readlines()):
                    base_line_sp = line.strip().split(" ")
                    img_path = base_line_sp[0]
                    if img_path not in source:
                        skip_imgs.append([index, img_path])
                        continue
                    src_lines = source[img_path]
                    # src_line_sp = src_line.strip().split(" ")
                    # src_box = [float(x) for x in src_line_sp[src_box_pos: src_box_pos + 4]]
                    base_box = list(map(float, base_line_sp[src_box_pos:src_box_pos + 4]))  # x, y, w, h
                    match_line = match_max_iou(base_box, src_lines, thd=thd, is_contain_imgwh=is_contain_imgwh)
                    if match_line is not None:
                        fw.write(match_line)
                    else:
                        if verbose:
                            print("no box match {}, box:{}".format(img_path, " ".join(list(map(str, base_box)))))
    else:
        base = OrderedDict()
        with open(base_path, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                line_sp = line.strip().split()
                img_path = line_sp[0]
                if img_path not in base:
                    base[img_path] = [line]
                else:
                    base[img_path].append(line)
        with open(dst_path, "w", encoding="utf-8") as fw:
            for index, img_path in enumerate(base.keys()):
                if img_path not in source:
                    skip_imgs.append([index, img_path])
                    continue
                for line in source[img_path]:
                    fw.write(line)

    if len(skip_imgs):
        print("base_path:{}\nsrc_path:{}".format(base_path, src_path))
        print("The img_path in base_path, not in src_path number {}".format(len(skip_imgs)))
        if verbose:
            import pprint

            pprint.pprint(skip_imgs)


def gen_FL(src_path, base_path, dst_path):
    source = {}
    with open(base_path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            img_name = get_unique_name(img_path)
            source[img_name] = line
    cnt = 0
    with open(src_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for index, line in enumerate(fr.readlines()):
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                img_name = get_unique_name(img_path)
                if img_name not in source:
                    fw.write(line)
                else:
                    cnt += 1
    print("cnt {}".format(cnt))


def modify_bbox_info(src_path, base_path, dst_path, skip_boxes_path_txt=None, ignore_imgs_path_txt=None,
                     use_kpt_box=False, thd=0.3, verbose=False):
    """dst_path的显示顺序和base_path的一致, 修改base_path中的bbox内容，其内容来自于src_path，如果src_path中不存在，则放弃该行

    Args:
        src_path:
        base_path:
        dst_path:

    Returns:

    """
    source = {}
    det_box_num = 0
    with open(src_path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            img_name = get_unique_name(img_path)
            det_box_num += 1
            if img_name not in source:
                source[img_name] = [line]
            else:
                source[img_name].append(line)
    src_box_pos = 2
    dst_box_pos = 2
    skip_boxes = []
    skip_imgs_set = set()
    ignore_imgs = []
    ignore_imgs_set = set()
    all_imgs = set()
    baseinfo = {}
    iou_ignore_cnt = 0
    cnt = 0
    with open(base_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for index, line in enumerate(fr.readlines()):
                base_line_sp = line.strip().split(" ")
                img_path = base_line_sp[0]
                img_name = get_unique_name(img_path)
                # if img_name.endswith("7895237528_8"):
                if img_name.endswith("rqAW55tAAIzXHVOing534_bbox1"):
                    print()
                if img_name not in baseinfo:
                    baseinfo[img_name] = [line]
                else:
                    baseinfo[img_name].append(line)
                all_imgs.add(img_path)
                if img_name not in source:
                    skip_boxes.append([index, img_path])
                    skip_imgs_set.add(img_path)
                    continue
                src_lines = source[img_name]
                ignore = []
                if use_kpt_box:
                    base_kpt = list(map(float, base_line_sp[src_box_pos + 4:src_box_pos + 4 + 10]))  # x, y
                    base_box = get_enclose_bbox_of_kpt(base_kpt)
                else:
                    base_box = list(map(float, base_line_sp[src_box_pos:src_box_pos + 4]))  # x, y, w, h
                match_line = match_max_iou(base_box, src_lines, thd=thd)
                if match_line is not None:
                    cnt += 1
                    src_line_sp = match_line.strip().split(" ")
                    new_base_line_sp = base_line_sp[:dst_box_pos] + src_line_sp[src_box_pos:src_box_pos + 4] + \
                                       base_line_sp[dst_box_pos + 4:]
                    new_line = " ".join(new_base_line_sp) + "\n"
                    fw.write(new_line)
                else:
                    iou_ignore_cnt += 1
                    ignore_imgs.append([index, img_path, ignore])
                    ignore_imgs_set.add(img_path)
                # for src_line in src_lines:
                #     src_line_sp = src_line.strip().split(" ")
                #     src_box = list(map(float, src_line_sp[src_box_pos:src_box_pos + 4]))  # x, y, w, h
                #     base_box = list(map(float, base_line_sp[src_box_pos:src_box_pos + 4]))  # x, y, w, h
                #     iou = IOU_XYWH(src_box, base_box)
                #     ignore.append(iou)
                #     if iou < 0.3:
                #         continue
                #     cnt += 1
                #     new_base_line_sp = base_line_sp[:dst_box_pos] + src_line_sp[src_box_pos:src_box_pos + 4] + \
                #                        base_line_sp[dst_box_pos + 4:]
                #     new_line = " ".join(new_base_line_sp) + "\n"
                #     fw.write(new_line)
                # # if cnt >= 2:
                # #     print(new_base_line_sp, ignore)
                # if cnt == 0:
                #     ignore_imgs.append([index, img_path, ignore])
                #     ignore_imgs_set.add(img_path)

    if len(skip_boxes):
        if verbose:
            import pprint
            print("skip {} box".format(len(skip_boxes)))
            pprint.pprint(skip_boxes[:10])
        if skip_boxes_path_txt is not None:
            os.makedirs(os.path.dirname(skip_boxes_path_txt), exist_ok=True)
            with open(skip_boxes_path_txt, "w") as fw1:
                for img_path in skip_imgs_set:
                    fw1.write(img_path + " 0 0\n")

    if len(ignore_imgs):
        if verbose:
            import pprint
            print("ignore {} box".format(len(ignore_imgs)))
            pprint.pprint(ignore_imgs[:10])
        if ignore_imgs_path_txt is not None:
            os.makedirs(os.path.dirname(ignore_imgs_path_txt), exist_ok=True)
            with open(ignore_imgs_path_txt, "w") as fw2:
                for img_path in ignore_imgs_set:
                    img_name = get_unique_name(img_path)
                    src_lines = source[img_name]
                    for src_line in src_lines:
                        fw2.write(src_line)
                    base_lines = baseinfo[img_name]
                    for base_line in base_lines:
                        fw2.write(base_line)
    remain_box_num = cnt
    remain_image_num = len(all_imgs) - len(skip_imgs_set) - len(ignore_imgs_set)
    print(
        "The img_path in base_path, skip box: {} total box:{},remain_box:{} skip img:{} total image:{} remain image:{}".format(
            len(skip_boxes), det_box_num, remain_box_num, len(skip_imgs_set), len(all_imgs), remain_image_num))
    print(
        "The img_path in src_path, all iou < {}, ignore box:{}, ignore image:{} total image:{}, remain image:{}".format(
            thd, len(ignore_imgs), len(ignore_imgs_set), len(all_imgs), remain_image_num))
    return remain_box_num, remain_image_num


def modify_bbox_and_kpt_info(src_path, base_path, dst_path, skip_boxes_path_txt=None, ignore_imgs_path_txt=None,
                             use_kpt_box=False, thd=0.3, verbose=False):
    """dst_path的显示顺序和base_path的一致, 修改base_path中的bbox和kpt内容，其内容来自于src_path，如果src_path中不存在，则放弃该行

    Args:
        src_path:
        base_path:
        dst_path:

    Returns:

    """
    source = {}
    det_box_num = 0
    with open(src_path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            img_name = get_unique_name(img_path)
            det_box_num += 1
            if img_name not in source:
                source[img_name] = [line]
            else:
                source[img_name].append(line)
    src_box_pos = 2
    dst_box_pos = 2
    skip_boxes = []
    skip_imgs_set = set()
    ignore_imgs = []
    ignore_imgs_set = set()
    all_imgs = set()
    baseinfo = {}
    iou_ignore_cnt = 0
    cnt = 0
    with open(base_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for index, line in enumerate(fr.readlines()):
                base_line_sp = line.strip().split(" ")
                img_path = base_line_sp[0]
                img_name = get_unique_name(img_path)
                # if img_name.endswith("7895237528_8"):
                if img_name.endswith("rqAW55tAAIzXHVOing534_bbox1"):
                    print()
                if img_name not in baseinfo:
                    baseinfo[img_name] = [line]
                else:
                    baseinfo[img_name].append(line)
                all_imgs.add(img_path)
                if img_name not in source:
                    skip_boxes.append([index, img_path])
                    skip_imgs_set.add(img_path)
                    continue
                src_lines = source[img_name]
                ignore = []
                if use_kpt_box:
                    base_kpt = list(map(float, base_line_sp[src_box_pos + 4:src_box_pos + 4 + 10]))  # x, y
                    base_box = get_enclose_bbox_of_kpt(base_kpt)
                else:
                    base_box = list(map(float, base_line_sp[src_box_pos:src_box_pos + 4]))  # x, y, w, h
                match_line = match_max_iou(base_box, src_lines, thd=thd)
                if match_line is not None:
                    cnt += 1
                    src_line_sp = match_line.strip().split(" ")
                    new_base_line_sp = base_line_sp[:dst_box_pos] + src_line_sp[src_box_pos:src_box_pos + 14]
                    new_line = " ".join(new_base_line_sp) + "\n"
                    fw.write(new_line)
                else:
                    iou_ignore_cnt += 1
                    ignore_imgs.append([index, img_path, ignore])
                    ignore_imgs_set.add(img_path)
                # for src_line in src_lines:
                #     src_line_sp = src_line.strip().split(" ")
                #     src_box = list(map(float, src_line_sp[src_box_pos:src_box_pos + 4]))  # x, y, w, h
                #     base_box = list(map(float, base_line_sp[src_box_pos:src_box_pos + 4]))  # x, y, w, h
                #     iou = IOU_XYWH(src_box, base_box)
                #     ignore.append(iou)
                #     if iou < 0.3:
                #         continue
                #     cnt += 1
                #     new_base_line_sp = base_line_sp[:dst_box_pos] + src_line_sp[src_box_pos:src_box_pos + 4] + \
                #                        base_line_sp[dst_box_pos + 4:]
                #     new_line = " ".join(new_base_line_sp) + "\n"
                #     fw.write(new_line)
                # # if cnt >= 2:
                # #     print(new_base_line_sp, ignore)
                # if cnt == 0:
                #     ignore_imgs.append([index, img_path, ignore])
                #     ignore_imgs_set.add(img_path)

    if len(skip_boxes):
        if verbose:
            import pprint
            print("skip {} box".format(len(skip_boxes)))
            pprint.pprint(skip_boxes[:10])
        if skip_boxes_path_txt is not None:
            os.makedirs(os.path.dirname(skip_boxes_path_txt), exist_ok=True)
            with open(skip_boxes_path_txt, "w") as fw1:
                for img_path in skip_imgs_set:
                    fw1.write(img_path + " 0 0\n")

    if len(ignore_imgs):
        if verbose:
            import pprint
            print("ignore {} box".format(len(ignore_imgs)))
            pprint.pprint(ignore_imgs[:10])
        if ignore_imgs_path_txt is not None:
            os.makedirs(os.path.dirname(ignore_imgs_path_txt), exist_ok=True)
            with open(ignore_imgs_path_txt, "w") as fw2:
                for img_path in ignore_imgs_set:
                    img_name = get_unique_name(img_path)
                    src_lines = source[img_name]
                    for src_line in src_lines:
                        fw2.write(src_line)
                    base_lines = baseinfo[img_name]
                    for base_line in base_lines:
                        fw2.write(base_line)
    remain_box_num = cnt
    remain_image_num = len(all_imgs) - len(skip_imgs_set) - len(ignore_imgs_set)
    print(
        "The img_path in base_path, skip box: {} total box:{},remain_box:{} skip img:{} total image:{} remain image:{}".format(
            len(skip_boxes), det_box_num, remain_box_num, len(skip_imgs_set), len(all_imgs), remain_image_num))
    print(
        "The img_path in src_path, all iou < {}, ignore box:{}, ignore image:{} total image:{}, remain image:{}".format(
            thd, len(ignore_imgs), len(ignore_imgs_set), len(all_imgs), remain_image_num))
    return remain_box_num, remain_image_num


def shuffl_seq(src_path, dst_path):
    print("shuffle seq")
    import random
    random.seed(1234)
    total = []
    with open(src_path, "r") as fr:
        for line in fr.readlines():
            total.append(line)
    random.shuffle(total)
    with open(dst_path, "w") as fw:
        for line in total:
            fw.write(line)


def modify_kpt_info(src_path, base_path, dst_path, skip_boxes_path_txt, ignore_imgs_path_txt):
    """dst_path的显示顺序和base_path的一致, 修改base_path中的kpt内容，其内容来自于src_path，如果src_path中不存在，则放弃该行

    Args:
        src_path:
        base_path:
        dst_path:

    Returns:

    """
    source = {}
    det_box_num = 0
    with open(src_path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            img_name = get_unique_name(img_path)
            det_box_num += 1
            if img_name not in source:
                source[img_name] = [line]
            else:
                source[img_name].append(line)
    src_box_pos = 2
    dst_box_pos = 2
    skip_boxes = []
    skip_imgs_set = set()
    ignore_imgs = []
    ignore_imgs_set = set()
    all_imgs = set()
    baseinfo = {}
    with open(base_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for index, line in enumerate(fr.readlines()):
                base_line_sp = line.strip().split(" ")
                img_path = base_line_sp[0]
                img_name = get_unique_name(img_path)
                if img_name not in baseinfo:
                    baseinfo[img_name] = [line]
                else:
                    baseinfo[img_name].append(line)
                all_imgs.add(img_path)
                if img_name not in source:
                    skip_boxes.append([index, img_path])
                    skip_imgs_set.add(img_path)
                    continue
                src_lines = source[img_name]
                ignore = []
                base_box = list(map(float, base_line_sp[src_box_pos:src_box_pos + 4]))  # x, y, w, h
                match_line = match_max_iou(base_box, src_lines, thd=0.3)
                if match_line is not None:
                    new_base_line_sp = base_line_sp[:src_box_pos + 4] + match_line[dst_box_pos + 4:]
                    new_line = " ".join(new_base_line_sp) + "\n"
                    fw.write(new_line)
                else:
                    ignore_imgs.append([index, img_path, ignore])
                    ignore_imgs_set.add(img_path)

    if len(skip_boxes):
        import pprint
        # pprint.pprint(skip_boxes)
        # print("......")
        os.makedirs(os.path.dirname(skip_boxes_path_txt), exist_ok=True)
        with open(skip_boxes_path_txt, "w") as fw1:
            for img_path in skip_imgs_set:
                fw1.write(img_path + " 0 0\n")
    print("ignore")
    if len(ignore_imgs):
        import pprint
        pprint.pprint(ignore_imgs)
        os.makedirs(os.path.dirname(ignore_imgs_path_txt), exist_ok=True)
        with open(ignore_imgs_path_txt, "w") as fw2:
            for img_path in ignore_imgs_set:
                src_lines = source[img_path]
                for src_line in src_lines:
                    fw2.write(src_line)
                base_lines = baseinfo[img_path]
                for base_line in base_lines:
                    fw2.write(base_line)

    print("The img_path in base_path, skip box: {} total box:{}, skip img:{} total image:{}".format(
        len(skip_boxes), det_box_num, len(skip_imgs_set), len(all_imgs)))
    print("The img_path in src_path, but all iou < 0.3, ignore box number {} ignore image:{} total box image:{}".format(
        len(ignore_imgs), len(ignore_imgs_set), len(all_imgs) - len(skip_imgs_set)))


def select_by_name(src_path, base_path, dst_path, src_root=None):
    """dst_path的显示顺序和base_path的一致, 如果src_path中的图片名字在base_path中出现则将src_path中该条数据存储到dst_path，否则删除

    Args:
        src_path:
        base_path:
        dst_path:
    """
    base = {}
    det_box_num = 0
    with open(base_path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            img_name = get_unique_name(img_path)
            det_box_num += 1
            if img_name not in base:
                base[img_name] = [line]
            else:
                base[img_name].append(line)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for index, line in enumerate(fr.readlines()):
                src_line_sp = line.strip().split(" ")
                img_path = src_line_sp[0]
                img_name = get_unique_name(img_path, src_root)
                name_in_source = [1 if img_name in name else 0 for name in base.keys()]
                if sum(name_in_source) >= 1:
                    fw.write(line)


def modify_seq_align(src_path, base_path, dst_path):
    """dst_path的显示顺序和base_path的一致,，内容来自于src_path

    Args:
        src_path:
        base_path:
        dst_path:
    """
    source = {}
    with open(src_path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            img_name = get_unique_name(img_path, isaligin=True)
            if img_name in source:
                raise Exception(img_name)
            source[img_name] = line
    src_box_pos = 2
    skip_imgs = []
    with open(base_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for index, line in enumerate(fr.readlines()):
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                img_name = get_unique_name(img_path, isaligin=True)
                if img_name not in source:
                    skip_imgs.append([index, img_path])
                    continue
                src_line = source[img_name]
                # src_line_sp = src_line.strip().split(" ")
                # src_box = [float(x) for x in src_line_sp[src_box_pos: src_box_pos + 4]]
                fw.write(src_line)
    if len(skip_imgs):
        print("The img_path in base_path, not in src_path number {}".format(len(skip_imgs)))
        import pprint
        pprint.pprint(skip_imgs)


def cwface_label_to_origin_lst(src_path, dst_path, verbose=False):
    if verbose:
        print("cwface label to origin lst")
    total = set()
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cnt = 0
    img_set = set()
    with open(src_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                if img_path in total:
                    img_set.add(img_path)
                    continue
                else:
                    total.add(img_path)
                    fw.write(line)
    if verbose:
        print("the same image path number {}".format(len(img_set)))


def select_by_boxiou(src_path, base_path, dst_path, thd=0.5):
    """dst_path的显示顺序和base_path的一致, 修改base_path中的bbox内容，其内容来自于src_path，如果src_path中不存在，则放弃该行

    Args:
        src_path:
        base_path:
        dst_path:

    Returns:

    """
    source = {}
    det_box_num = 0
    with open(src_path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            det_box_num += 1
            if img_path not in source:
                source[img_path] = [line]
            else:
                source[img_path].append(line)
    skip_imgs = []
    src_box_pos = 2
    with open(base_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for line in fr.readlines():
                base_line_sp = line.strip().split(" ")
                img_path = base_line_sp[0]
                if img_path not in source:
                    skip_imgs.append([img_path])
                    continue
                src_lines = source[img_path]
                # src_line_sp = src_line.strip().split(" ")
                # src_box = [float(x) for x in src_line_sp[src_box_pos: src_box_pos + 4]]
                base_box = list(map(float, base_line_sp[src_box_pos:src_box_pos + 4]))  # x, y, w, h
                match_line = match_max_iou(base_box, src_lines, thd=thd)
                if match_line is not None:
                    fw.write(match_line)


def gen_test_from_train(train_path, test_path, new_train_path, select_num=200):
    """dst_path的显示顺序和base_path的一致,，内容来自于src_path

    Args:
        src_path:
        base_path:
        dst_path:
    """
    import random
    from collections import defaultdict
    source = defaultdict(list)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    os.makedirs(os.path.dirname(new_train_path), exist_ok=True)
    with open(train_path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            img_name = get_unique_name(img_path, isaligin=False)
            source[img_name].append(line)
    select_img_name = []
    for img_name in source.keys():
        if len(source[img_name]) <= 1:
            select_img_name.append(img_name)
    random.seed(1234)
    random.shuffle(select_img_name)

    select_img_name = set(select_img_name[:select_num])
    with open(train_path, "r") as fr:
        with open(test_path, "w") as fw1:
            with open(new_train_path, "w") as fw2:
                for line in fr.readlines():
                    line_sp = line.strip().split(" ")
                    img_path = line_sp[0]
                    img_name = get_unique_name(img_path, isaligin=False)
                    if img_name in select_img_name:
                        fw1.write(line)
                    else:
                        fw2.write(line)


def remove_prefix(path, prefix):
    if prefix.endswith("/"):
        l = len(prefix)
    else:
        l = len(prefix) + 1
    path = path[l:]
    return path


def copy_or_mv_img_to_dir(src_path, src_img_root, dst_path, dst_img_root, copy=True):
    print("copy_or_mv_img_to_dir")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r") as fr:
        with open(dst_path, "w") as fw:
            for line in fr.readlines():
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                img_name = remove_prefix(img_path, src_img_root)
                dst_img_path = os.path.join(dst_img_root, img_name)
                os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
                if copy:
                    shutil.copy(img_path, dst_img_path)
                else:
                    shutil.move(img_path, dst_img_path)
                src_img_txt = os.path.splitext(img_path)[0] + ".txt"
                if os.path.exists(src_img_txt):
                    dst_img_txt = os.path.splitext(dst_img_path)[0] + ".txt"
                    if copy:
                        shutil.copy(src_img_txt, dst_img_txt)
                    else:
                        shutil.move(src_img_txt, dst_img_txt)
                new_line_sp = [dst_img_path] + line_sp[1:]
                new_line = " ".join(new_line_sp) + "\n"
                fw.write(new_line)


if __name__ == '__main__':
    # origin_img_dir = "/ssd/128x128/tang.v8.st/shenyang.orig/mask_dy_draw"
    # low_qual_path = "/ssd/128x128/tang.v8.st/shenyang.orig/fiq/filter.out.view.lst"
    # save_low_qual_orign_path = "/dataset/dataset/ssd/kpt/cluster_rst/sy.ssd.filter.draw.list"
    # align_list_to_orign_list(low_qual_path, save_low_qual_orign_path, origin_img_dir)

    # origin_list = "/dataset/dataset/ssd/kpt/cluster_rst/sy.ssd.filter0.draw.list"
    # det_list_src = "/ssd/128x128/tang.v8.st/shenyang.orig/det.ssd.list"
    # det_list_dst = "/ssd/128x128/tang.v8.st/shenyang.orig/det.ssd.filter0.list"
    # orign_list_to_det_list(origin_list, det_list_src, det_list_dst)

    # src_path = "/dataset/dataset/ssd/kpt_test_list/shenyang500/xc.zd.list"
    # det_label_path = "/dataset/dataset/ssd/128x128/zd/shenyang5w.orig/shenyang5w.det.list"
    # save_tang_label_path = "/dataset/dataset/ssd/kpt_test_list/shenyang500/shenyang500.tang.txt"
    # gen_shenyang500(src_path, det_label_path, save_tang_label_path)

    # src_tang_list = "/dataset/dataset/ssd/kpt_test_list/shenyang500/shenyang500.tang.txt"
    # src_draw_lis = "/dataset/dataset/ssd/kpt/cluster_rst/shenyang5w.ssd.draw.list"
    # dst_draw_list = "/dataset/dataset/ssd/kpt/cluster_rst/shenyang500.draw.list"
    # tang_list_to_draw_list(src_tang_list, src_draw_lis, dst_draw_list)

    # root_dir = "/ssd/data/workspace/kpt_test_img"
    # save_path = "/dataset/dataset/ssd/kpt/cluster_rst/kpt_test_img.txt"
    # gen_list_from_dir(root_dir, save_path)

    # src_path = "/dataset/dataset/ssd/kpt_test_list/SY_test/det/det.lst"
    # dst_path = "/dataset/dataset/ssd/kpt_test_list/SY_test/SY_test.tang.txt"
    # cwface_label_to_tang_label(src_path, dst_path)

    # filter_by_fiq()

    # src_path = "/dataset/dataset/ssd/kpt_train_list/shenyang_pose_2/gt2.structure_rect.cwface.txt"
    # dst_path = "/dataset/dataset/ssd/kpt_train_list/shenyang_pose_2/8k.gt2.structure_rect.cwface.txt"
    # select_some_sample(src_path, dst_path, num=8000, base=3)

    # src_path = "/dataset/dataset/ssd/kpt_data/face_data/labels/refine_28w_label_path.txt"
    # dst_path = "/dataset/dataset/ssd/kpt_train_list/28w/xc.cwface.txt"
    # tang_label_to_cwface_label(src_path, dst_path)

    # src_path = "/dataset/dataset/ssd/kpt_data/300W/d300W_landmarks_lbl_test.txt"
    # dst_path = "/dataset/dataset/ssd/kpt_test_list/300W/gt.cwface.txt"
    # delete_some_col(src_path, dst_path)

    # src_path = "/dataset/dataset/ssd/gesture/jiashicang/resize/txt/pred.jsc220713.gt.txt"
    # base_path = "/dataset/dataset/ssd/gesture/jiashicang/resize/txt/jsc220713.gt.txt"
    # dst_path = "/dataset/dataset/ssd/gesture/jiashicang/resize/txt/jsc220713.box.match.txt"
    # modify_seq(src_path, base_path, dst_path, is_use_box=True, verbose=True)
    # shutil.move(dst_path, src_path)

    # src_path = "/dataset/dataset/ssd/kpt_train_list/fuling/det/det.lst"
    # base_path = "/dataset/dataset/ssd/kpt_test_list/FL/gt.cwface.txt"
    # dst_path = "/dataset/dataset/ssd/kpt_train_list/fuling/det/det.lst1"
    # gen_FL(src_path, base_path, dst_path)

    # src_path = "/dataset/dataset/ssd/kpt_train_list/shenyang_pose_2/gt.structure_rect.fk.cwface.txt1"
    # base_path = "/dataset/dataset/ssd/kpt_train_list/shenyang_pose_2/gt2.structure_rect.cwface.txt"
    # dst_path = "/dataset/dataset/ssd/kpt_train_list/shenyang_pose_2/gt.structure_rect.fk.cwface.txt"
    # skip_imgs_path_txt = "/dataset/dataset/ssd/kpt_train_list/shenyang_pose_2/det_error/structure_rect_noface.gt.fk.txt"
    # ignore_imgs_path_txt = "/dataset/dataset/ssd/kpt_train_list/shenyang_pose_2/det_error/structure_rect_ignore.gt.fk.txt"
    # modify_kpt_info(src_path, base_path, dst_path, skip_imgs_path_txt, ignore_imgs_path_txt)
    # golabel_txt = "/dataset/dataset/ssd/kpt/cluster_rst/structure_detect_noface_JD.txt"
    # shutil.copyfile(skip_imgs_path_txt, golabel_txt)

    src_path = "/dataset/dataset/ssd/gesture/jiashicang/txtv2/jsc220713.gt.txt"
    base_path = "/data/gesture/jiashicang/resize/txtv2/test.jsc220713.gt.txt"
    dst_path = "/dataset/dataset/ssd/gesture/jiashicang/txtv2/test.jsc220713.gt.txt"
    src_root = "/dataset/dataset/ssd/gesture/jiashicang/"
    select_by_name(src_path, base_path, dst_path, src_root=src_root)

    # print("visualize")
    # save_draw_img_root = "/dataset/dataset/ssd/kpt_data/draw/profile200/structure_rect_gt2/imgs"
    # save_txt_path = "/dataset/dataset/ssd/kpt_test_list/profile200/draw/structure_rect.gt2.txt"
    # draw_box_kpt_from_cwface_path_same_image(dst_path, save_draw_img_root, save_txt_path)
    # save_txt_path2 = "/dataset/dataset/ssd/kpt/cluster_rst/profile200.draw.structure_rect.gt2.txt"
    # shutil.copyfile(save_txt_path, save_txt_path2)

    # src_path = "/dataset/dataset/ssd/kpt_test_list/SY.YAW/det_structure_rect/det.lst"
    # base_path = "/dataset/dataset/ssd/kpt_test_list/SY.YAW/gt.official.txt"
    # dst_path = "/dataset/dataset/ssd/kpt_test_list/SY.YAW/gt.structure_rect.cwface.txt1"
    # skip_imgs_path_txt = "/dataset/dataset/ssd/kpt_test_list/SY.YAW/det_error/structure_rect_noface.txt"
    # ignore_imgs_path_txt = "/dataset/dataset/ssd/kpt_test_list/SY.YAW/det_error/structure_rect_ignore.txt"
    # modify_bbox_info(src_path, base_path, dst_path, skip_imgs_path_txt, ignore_imgs_path_txt)

    # src_path = "/dataset/dataset/ssd/kpt_train_list/WIDER_train/gt.structure_rect.cwface.txt"
    # dst_path = "/dataset/dataset/ssd/kpt_train_list/WIDER_train/gt.structure_rect.cwface.txt1"
    # shuffl_seq(src_path, dst_path)

    # src_path = "/dataset/dataset/ssd/kpt_test_list/TRUNC/origin/origin.lst"
    # dst_path = "/dataset/dataset/ssd/kpt_test_list/TRUNC/origin/origin.lst1"
    # cwface_label_to_origin_lst(src_path, dst_path)
    # shutil.move(dst_path, src_path)

    # train_path = "/dataset/dataset/ssd/kpt_train_list/cd_overlap/gt.ai.bbox.structure_rect.fk.cwface.txt"
    # test_path = "/dataset/dataset/ssd/kpt_test_list/cd_overlap_test/gt.ai.bbox.structure_rect.fk.cwface.txt"
    # new_train_path = "/dataset/dataset/ssd/kpt_train_list/vgg2_train_overlap/gt.ai.bbox.structure_rect.fk.cwface.txt1"
    # select_num = 200
    # gen_test_from_train(train_path, test_path, new_train_path, select_num)
    # shutil.move(new_train_path, train_path)
    #
    # src_path = test_path
    # src_path = "/zhoudu/test/kpt_test/kpt220104b_175825/res_vis/structure_rect/epoch199/SY.YAW/kpt220104b_175825/retinaface320_bad_nme/rearange_true.txt"
    # src_img_root = "/dataset/dataset/ssd/kpt_data/shenyang_profile_2/"
    # src_img_root = "/opt/cwface-torch/all_data/SY.YAW"
    # dst_path = "/opt/cwface-torch/all_data/SY.YAW/gt.structure_rect.cwface.txt"
    # dst_img_root = "/opt/cwface-torch/all_data/SY.YAW/imgs"
    # copy = True
    # copy_or_mv_img_to_dir(src_path, src_img_root, dst_path, dst_img_root, copy)
