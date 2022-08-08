# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 10:53
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : gen_txt.py
# @Software: PyCharm
import os
from natsort import natsorted
import cv2
from multiprocessing import Pool
from utils.get_path_len import get_path_len
from utils.get_split_set import get_split_set


def traverse(src_dir, cls_idx=0, exts=None):
    if exts is None:
        exts = ['.jpg', '.png', '.bmp', ".jpeg"]
    labels = {}
    result = []
    cnt = -1
    for root, dirs, files in natsorted(os.walk(src_dir)):
        if len(files) > 0:
            class_name = root[get_path_len(src_dir):].split("/")[cls_idx]
            if class_name not in labels:
                cnt += 1
                labels[class_name] = str(cnt)
            for f in natsorted(files):
                ext = os.path.splitext(f)[-1].lower()
                if ext in exts:
                    result.append([os.path.join(root, f), str(cnt)])
    print(labels)  # key is cls_name value is the index
    return result, labels


def gen_txt(result, save_path, is_contain_imgwh=True):
    with open(save_path, "w", encoding="utf-8") as fw:
        for lines in result:
            img_path, class_name = lines
            if is_contain_imgwh:
                try:
                    img = cv2.imread(img_path)
                    img_h, img_w = img.shape[:2]
                    new_line = img_path + " " + class_name + " " + str(img_w) + " " + str(img_h) + "\n"
                except Exception:
                    print("can't open {}".format(img_path))
                    continue
            else:
                new_line = img_path + " " + class_name + "\n"
            # print(new_line)
            fw.write(new_line)
            fw.flush()


def main_multi_process(src_dir, save_path, pool_size, cls_idx=0, class_map_path=None, is_contain_imgwh=True):
    result, labels = traverse(src_dir, cls_idx)
    if class_map_path is not None:
        print("save_label_path: {}".format(class_map_path))
        os.makedirs(os.path.dirname(class_map_path), exist_ok=True)
        with open(class_map_path, "w", encoding="utf-8") as fw:
            for key, value in labels.items():  # key is the class name, value is the index
                line = value + " " + key + "\n"
                fw.write(line)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    split_set_list = get_split_set(len(result), pool_size)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("save_path: {}".format(save_path))
    pool = Pool(pool_size)

    for i in range(pool_size):
        save_path_i = save_path + str(i)
        start, end = split_set_list[i]
        src_data = result[start:end]
        result.append(pool.apply_async(gen_txt, args=(src_data, save_path_i, is_contain_imgwh)))
        # gen_txt(src_data, save_path_i, is_contain_imgwh)
        # deal(i, src_data, start, end, save_path_i, done)

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
    src_dirs = [
        "/dataset/dataset/ssd/gesture/leapGestRecog/imgs"
    ]
    save_paths = [
        "/dataset/dataset/ssd/gesture/leapGestRecog/txt/leapGestRecog.txt",
    ]
    class_map_paths = [
        "/dataset/dataset/ssd/gesture/leapGestRecog/txt/class_description.txt",
    ]
    # save_label_path = None
    is_contain_imgwh = True
    pool_size = 20
    cls_idx = 1
    for src_dir, save_path, class_map_path in zip(src_dirs, save_paths, class_map_paths):
        main_multi_process(src_dir, save_path, pool_size, cls_idx=cls_idx, class_map_path=class_map_path,
                           is_contain_imgwh=is_contain_imgwh)
        print("END")
