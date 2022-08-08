# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 11:37
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : save_crop_img.py
# @Software: PyCharm
import os
import cv2
import time
import shutil
from multiprocessing import Pool
from utils.box_and_kpt_utils import get_crop_img, get_resize_img
from utils.get_path_len import get_path_len
from utils.draw_box_kpt_utils import draw_box_and_kpt, show_img
from utils.get_split_set import get_split_set


def save_crop_img(i, src_lines, save_path, img_prefix, save_img_root, do_crop=True, len_thd=400, is_contain_imgwh=True,
                  vis=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    start = time.time()
    with open(save_path, "w", encoding="utf-8") as fw:
        for j, line in enumerate(src_lines):
            if j % 1000 == 0:
                print("pid {} save {} spend {} s".format(i, j, time.time() - start))
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            label = line_sp[1]
            img = cv2.imread(img_path)
            if is_contain_imgwh:
                img_w = line_sp[2]
                img_h = line_sp[3]
                box = line_sp[4:8]
            else:
                box = line_sp[2:6]
            box = list(map(float, box))
            kpt = None
            if vis:
                img_show = draw_box_and_kpt(img.copy(), box, box_color=(255, 0, 0),
                                            box_txt="gt:{}".format(label))
                show_img(img_show)
            if do_crop:
                img, box, kpt = get_crop_img(img, box, kpt, scale=0.5)
                if vis:
                    img_show = draw_box_and_kpt(img.copy(), box, box_color=(0, 255, 0),
                                                box_txt="gt:{}".format(label))
                    show_img(img_show)
            new_img_h, new_img_w, c = img.shape
            if max(new_img_h, new_img_w) > len_thd:
                img, box, kpt = get_resize_img(img, box, len_thd, len_thd, kpt=kpt)
                box = box[0]
                new_img_h, new_img_w, c = img.shape
                if vis:
                    img_show = draw_box_and_kpt(img.copy(), box, box_color=(0, 0, 255),
                                                box_txt="gt:{}".format(label))
                    show_img(img_show)
            box_str = " ".join(list(map(lambda x: "{:.2f}".format(x), box)))
            save_img_path = os.path.join(save_img_root, img_path[get_path_len(img_prefix):])
            os.makedirs(os.path.dirname(save_img_path).encode("utf-8"), exist_ok=True)
            cv2.imwrite(save_img_path, img)
            if is_contain_imgwh:
                new_line = save_img_path + " " + label + " " + str(new_img_w) + " " + str(
                    new_img_h) + " " + box_str + "\n"
            else:
                new_line = save_img_path + " " + label + " " + box_str + "\n"
            # print(new_line)
            fw.write(new_line)
            fw.flush()


def main_multi_process(src_path, save_path, img_prefix, save_img_root, pool_size, do_crop=True, len_thd=400):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pool = Pool(pool_size)
    sources = []
    with open(src_path, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            sources.append(line)
    split_set_list = get_split_set(len(sources), pool_size)
    result = []
    for i in range(pool_size):
        save_path_i = save_path + str(i)
        start, end = split_set_list[i]
        src_lines = sources[start:end]
        result.append(pool.apply_async(save_crop_img,
                                       args=(i, src_lines, save_path_i, img_prefix, save_img_root, do_crop, len_thd)))
        # save_crop_img(i, src_lines, save_path_i, img_prefix, save_img_root, do_crop, len_thd)
    pool.close()
    pool.join()
    with open(save_path, "w") as fw:
        for i in range(pool_size):
            save_path_i = save_path + str(i)
            with open(save_path_i, "r") as fr:
                for line in fr.readlines():
                    fw.write(line)
            os.remove(save_path_i)


if __name__ == '__main__':
    src_path = "/dataset/dataset/ssd/gesture/jiashicang/jsc220715_garage/txt/jsc220715_garage.box.txt"
    img_prefix = "/dataset/dataset/ssd/gesture/jiashicang/"

    save_path = "/dataset/dataset/ssd/gesture/jiashicang/resize/txtv2/jsc220715_garage.txt"
    save_img_root = "/dataset/dataset/ssd/gesture/jiashicang/resize/imgs"
    print(save_path)
    pool_size = 25
    do_crop = False
    len_thd = 416
    main_multi_process(src_path, save_path, img_prefix, save_img_root, pool_size, do_crop=do_crop, len_thd=len_thd)
    print("END")
