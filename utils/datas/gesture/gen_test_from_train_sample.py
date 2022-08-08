# -*- coding: utf-8 -*-
# @Time    : 2021/12/15 13:22
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : gen_test_from_train_sample.py
# @Software: PyCharm
import os, sys
from collections import defaultdict
import random


def gen_test_from_train(train_path, test_path, new_train_path=None, cls_num=None):
    """
    """
    id_to_path = defaultdict(list)
    source = defaultdict(list)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    if new_train_path is not None:
        os.makedirs(os.path.dirname(new_train_path), exist_ok=True)

    with open(train_path, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            img_id = line_sp[1]
            id_to_path[img_id].append(img_path)
            source[img_path].append(line)
    select_img_names = []
    random.seed(1234)
    train_num = {}
    test_num = {}
    for img_id in sorted(id_to_path.keys()):
        # if img_id != "0":  # 用于生成负例数据
        #     continue
        img_names = id_to_path[img_id]
        random.shuffle(img_names)
        select_num = min(int(len(img_names) * 0.1), 50)
        # select_num = 200  # 用于生成负例数据
        train_num[img_id] = (len(img_names) - select_num)
        test_num[img_id] = select_num
        select_img_names.extend(img_names[:select_num])
    print("train num:")
    total_train_num = 0
    for img_id in range(0, cls_num):
        img_id = str(img_id)
        n = train_num[img_id] if img_id in train_num else 0
        # print("{}/{}".format(img_id, n))
        print("{}".format(n))
        total_train_num += n
    print("test num:")
    total_test_num = 0
    for img_id in range(0, cls_num):
        img_id = str(img_id)
        n = test_num[img_id] if img_id in test_num else 0
        # print("{}/{}".format(img_id, n))
        print("{}".format(n))
        total_test_num += n
    print("select train num: {}, test num: {}, class num: {}".format(total_train_num, total_test_num, len(id_to_path)))
    select_img_names = set(select_img_names)
    with open(train_path, "r", encoding="utf-8") as fr:
        with open(test_path, "w", encoding="utf-8") as fw1:
            if new_train_path is not None:
                with open(new_train_path, "w", encoding="utf-8") as fw2:
                    for line in fr.readlines():
                        line_sp = line.strip().split(" ")
                        img_path = line_sp[0]
                        if img_path in select_img_names:
                            fw1.write(line)
                            fw1.flush()
                        else:
                            fw2.write(line)
                            fw2.flush()
            else:
                for line in fr.readlines():
                    line_sp = line.strip().split(" ")
                    img_path = line_sp[0]
                    if img_path in select_img_names:
                        fw1.write(line)
                        fw1.flush()


if __name__ == '__main__':
    cls_name = "c9"
    cls_num = 9
    datas = {
        "ges30_cloudwalk":
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/ges30_cloudwalk.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/test.ges30_cloudwalk.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/train.ges30_cloudwalk.map.box.txt",
            },
        "ges30_imgs_1015_1017":
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/ges30_imgs_1015_1017.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/test.ges30_imgs_1015_1017.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/train.ges30_imgs_1015_1017.map.box.txt",
            },
        "ges30_imgs_1018_1019":
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/ges30_imgs_1018_1019.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/test.ges30_imgs_1018_1019.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/train.ges30_imgs_1018_1019.map.box.txt",
            },
        "neg_ges30_imgs_1018_1019":
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txt/ges30_imgs_1018_1019.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v3/neg.test.ges30_imgs_1018_1019.map.box.txt",
                "new_train_path": None
            },
        "pos_ges30_cloudwalk.ges30_imgs_1015_1017":
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v2/train.ges30_cloudwalk.ges30_imgs_1015_1017.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v2/pos.txt",
                "new_train_path": None
            },
        "indoor_multi_scenario":
            {
                "train_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/cls7/indoor_multi_scenario.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/cls7/test.indoor_multi_scenario.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/cls7/train.indoor_multi_scenario.map.box.txt",
            },
        "indoor_office":
            {
                "train_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/cls7/indoor_office.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/cls7/test.indoor_office.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/cls7/train.indoor_office.map.box.txt",
            },
        "ges30_cloudwalk_digit":
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/digit/ges30_cloudwalk_digit.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/digit/test.ges30_cloudwalk_digit.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/digit/train.ges30_cloudwalk_digit.map.box.txt",
            },
        "ges30_imgs_1015_1017_digit":
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/digit/ges30_imgs_1015_1017_digit.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/digit/test.ges30_imgs_1015_1017_digit.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/digit/train.ges30_imgs_1015_1017_digit.map.box.txt",
            },
        "ges30_imgs_1018_1019_digit":
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/digit/ges30_imgs_1018_1019_digit.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/digit/test.ges30_imgs_1018_1019_digit.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/digit/train.ges30_imgs_1018_1019_digit.map.box.txt",
            },
        "indoor_multi_scenario_digit":
            {
                "train_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/digit/indoor_multi_scenario_digit.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/digit/test.indoor_multi_scenario_digit.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/digit/train.indoor_multi_scenario_digit.map.box.txt",
            },
        "indoor_office_digit":
            {
                "train_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/digit/indoor_office_digit.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/digit/test.indoor_office_digit.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/digit/train.indoor_office_digit.map.box.txt",
            },
        "ges30_cloudwalk_cls13":
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/ges30_cloudwalk_cls13.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/test.ges30_cloudwalk_cls13.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/train.ges30_cloudwalk_cls13.map.box.txt",
            },
        "ges30_imgs_1015_1017_cls13":
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/ges30_imgs_1015_1017_cls13.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/test.ges30_imgs_1015_1017_cls13.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/train.ges30_imgs_1015_1017_cls13.map.box.txt",
            },
        "ges30_imgs_1018_1019_cls13":
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/ges30_imgs_1018_1019_cls13.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/test.ges30_imgs_1018_1019_cls13.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/train.ges30_imgs_1018_1019_cls13.map.box.txt",
            },
        "indoor_multi_scenario_cls13":
            {
                "train_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/cls13/indoor_multi_scenario_cls13.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/cls13/test.indoor_multi_scenario_cls13.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/cls13/train.indoor_multi_scenario_cls13.map.box.txt",
            },
        "indoor_office_cls13":
            {
                "train_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/cls13/indoor_office_cls13.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/cls13/test.indoor_office_cls13.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/cls13/train.indoor_office_cls13.map.box.txt",
            },
        "ges30_cloudwalk_" + cls_name:
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/ges30_cloudwalk.map.box.match.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/test.ges30_cloudwalk.map.box.match.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/train.ges30_cloudwalk.map.box.match.txt",
            },
        "ges30_imgs_1015_1017_" + cls_name:
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/ges30_imgs_1015_1017.map.box.match.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/test.ges30_imgs_1015_1017.map.box.match.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/train.ges30_imgs_1015_1017.map.box.match.txt",
            },
        "ges30_imgs_1018_1019_" + cls_name:
            {
                "train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/ges30_imgs_1018_1019.map.box.match.txt",
                "test_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/test.ges30_imgs_1018_1019.map.box.match.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/train.ges30_imgs_1018_1019.map.box.match.txt",
            },
        "indoor_multi_scenario_" + cls_name:
            {
                "train_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/" + cls_name + "/indoor_multi_scenario.map.box.match.txt",
                "test_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/" + cls_name + "/test.indoor_multi_scenario.map.box.match.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/" + cls_name + "/train.indoor_multi_scenario.map.box.match.txt",
            },
        "indoor_office_" + cls_name:
            {
                "train_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/" + cls_name + "/indoor_office.map.box.match.txt",
                "test_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/" + cls_name + "/test.indoor_office.map.box.match.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/" + cls_name + "/train.indoor_office.map.box.match.txt",
            },
        "hagrid_" + cls_name:
            {
                "train_path": "/dataset/dataset/ssd/gesture/hagrid/" + cls_name + "/hagrid.map.box.match.txt",
                "test_path": "/dataset/dataset/ssd/gesture/hagrid/" + cls_name + "/test.hagrid.map.box.match.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/hagrid/" + cls_name + "/train.hagrid.map.box.match.txt",
            },
        # "jsc220713_" + cls_name:
        #     {
        #         "train_path": "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/jsc220713.gt.txt",
        #         "test_path": "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/test.jsc220713.gt.txt",
        #         "new_train_path": "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/train.jsc220713.gt.txt",
        #     },
        "jsc220713_" + cls_name:
            {
                "train_path": "/dataset/dataset/ssd/gesture/jiashicang/resize/txtv2/jsc220713.gt.txt",
                "test_path": "/dataset/dataset/ssd/gesture/jiashicang/resize/txtv2/test.jsc220713.gt.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/jiashicang/resize/txtv2/train.jsc220713.gt.txt",
            },
        "jsc220718_day_" + cls_name:
            {
                "train_path": "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/jsc220718_day.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/test.jsc220718_day.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/train.jsc220718_day.map.box.txt",
            },
        "jsc220715_garage_" + cls_name:
            {
                "train_path": "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/jsc220715_garage.map.box.txt",
                "test_path": "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/test.jsc220715_garage.map.box.txt",
                "new_train_path": "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/train.jsc220715_garage.map.box.txt",
            },
    }
    # datanames = ["ges30_cloudwalk", "ges30_imgs_1015_1017", "ges30_imgs_1018_1019"]
    # datanames = ["ges30_cloudwalk_digit", "ges30_imgs_1015_1017_digit", "ges30_imgs_1018_1019_digit", "indoor_multi_scenario_digit", ]
    # datanames = ["indoor_multi_scenario_digit", ]
    datanames = ["ges30_cloudwalk_" + cls_name, "ges30_imgs_1015_1017_" + cls_name, "ges30_imgs_1018_1019_" + cls_name,
                 "indoor_multi_scenario_" + cls_name, "indoor_office_" + cls_name, "zptest_" + cls_name,
                 "jsc220713_" + cls_name, "jsc220718_day_" + cls_name, "jsc220715_garage_" + cls_name]
    datanames = ["jsc220715_garage_" + cls_name]
    for dataname in datanames:
        print("deal {}".format(dataname))
        train_path = datas[dataname]["train_path"]
        test_path = datas[dataname]["test_path"]
        new_train_path = datas[dataname]["new_train_path"]
        print("train_path: {}".format(train_path))
        print("test_path: {}".format(test_path))
        print("new_train_path: {}".format(new_train_path))
        gen_test_from_train(train_path, test_path, new_train_path, cls_num)
    print("END!")
