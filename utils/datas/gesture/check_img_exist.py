# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 18:40
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : check_img_exist.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 18:39
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : t.py
# @Software: PyCharm
import os, sys
from time import time
from multiprocessing import Pool
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))


def get_split_set(total_num, thread_num):
    avg_num = total_num // thread_num
    split_set_list = []

    for i in range(thread_num):
        start = i * avg_num
        end = (i + 1) * avg_num

        if i == thread_num - 1:
            end = total_num
        split_set_list.append((start, end))

    return split_set_list


def deal(i, src_data, start, end, save_path, done):
    print("pid{}:{} deal num:{} start:{} end:{}".format(i, os.getpid(), len(src_data), start, end))
    cnt = 0
    t0 = time()
    t = time()

    select_cnt = 0
    with open(save_path, "a+") as fw:
        for line in src_data:
            if cnt % 10000 == 0:
                print("pid{}:{} deal {}, select {} images, spends {} s, total {} s".format(
                    i, os.getpid(), cnt, select_cnt, time() - t, time() - t0))
                t = time()
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            cnt += 1
            if img_path in done:
                continue
            try:
                img = cv2.imread(img_path)
            except Exception as e:
                print("pid{}:{} path:{}".format(i, os.getpid(), img_path))
                fw.write(img_path + "\n")
                select_cnt += 1

    print("pid{}:{} select {} error images".format(i, os.getpid(), select_cnt))


def main_multi_process(src_path, save_path, pool_size, root=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pool = Pool(pool_size)
    result = []
    # 已经处理过的图片
    done = set()
    # if os.path.exists(save_path):
    #     with open(save_path, "r") as fr:
    #         for line in fr.readlines():
    #             line_sp = line.strip().split(" ")
    #             img_path = line_sp[0]
    #             done.add(img_path)
    sources = []
    with open(src_path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.split(" ")
            img_path = line_sp[0]
            if root is not None:
                sources.append(os.path.join(root, img_path))
            else:
                sources.append(img_path)
    split_set_list = get_split_set(len(sources), pool_size)

    for i in range(pool_size):
        save_path_i = save_path + str(i)
        start, end = split_set_list[i]
        src_data = sources[start:end]
        result.append(pool.apply_async(deal,
                                       args=(i, src_data, start, end, save_path_i, done)))
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
    src_path = "/dataset/dataset/ssd/gesture/jiashicang/resize/txtv2/test.jsc220713.gt.txt"
    save_path = "/dataset/dataset/ssd/gesture/jiashicang/resize/txtv2/wrong.txt"
    root = None

    pool_size = 4
    main_multi_process(src_path, save_path, pool_size, root=root)
    print("END")
