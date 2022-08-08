# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 16:47
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : gen_common_scenario_mobile_201021.py
# @Software: PyCharm
import os
from collections import defaultdict

label1_str = {}


def get_map_txt(src_map_path, dst_map_path, src_path, dst_path):
    with open(src_map_path, "r", encoding="utf-8") as fr1:
        for line1 in fr1.readlines():
            line1_sp = line1.strip().split(" ")
            label1_str[line1_sp[0]] = line1_sp[1]  # key: index, value: name
    print("  src: key: index, value: name")
    print("  ", label1_str)
    label2_str = {}
    with open(dst_map_path, "r", encoding="utf-8") as fr2:
        for line2 in fr2.readlines():
            line2_sp = line2.strip().split(" ")
            # label_str.append("\"" + line2_sp[1] + "\"")
            label2_str[line2_sp[1]] = line2_sp[0]  # key: name, value: index
    print("  dst: key: name, value: index")
    print("  ", label2_str)
    # print([k for k, v in label2_str.items()])

    label_map = {}  # 将src的类别index映射到dst的类别index
    for k1, v1 in label1_str.items():
        for k2, v2 in label2_str.items():
            if v1 == k2:
                label_map[k1] = v2
    print("  map: key: src_index, value: dst_index")
    print("  ", label_map)

    lines = defaultdict(list)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            label = line_sp[1]
            if label not in label_map:
                continue
            new_label = label_map[label]
            line_sp[1] = new_label
            new_line = " ".join(line_sp) + "\n"
            lines[new_label].append(new_line)
    with open(dst_path, "w", encoding="utf-8") as fw:
        for new_label in sorted(lines.keys()):
            for new_line in lines[new_label]:
                fw.write(new_line)
                fw.flush()


if __name__ == '__main__':
    cls_name = "c9"
    datas = {
        "ges30_cloudwalk": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_cloudwalk.txt",
            "src_map_path": "desc/class_description.ges30_cloudwalk.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/ges30_cloudwalk.map.txt",
            "dst_map_path": "desc/class_description_cls7.txt"
        },
        "ges30_imgs_1015_1017": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_imgs_1015_1017.txt",
            "src_map_path": "desc/class_description.ges30_imgs_1015_1017.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/ges30_imgs_1015_1017.map.txt",
            "dst_map_path": "desc/class_description_cls7.txt"
        },
        "ges30_imgs_1018_1019": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_imgs_1018_1019.txt",
            "src_map_path": "desc/class_description.ges30_imgs_1018_1019.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/ges30_imgs_1018_1019.map.txt",
            "dst_map_path": "desc/class_description_cls7.txt"
        },
        "indoor_multi_scenario": {
            "src_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/txt/indoor_multi_scenario.box.txt",
            "src_map_path": "desc/class_description.indoor_multi_scenario.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/cls7/indoor_multi_scenario.map.box.txt",
            "dst_map_path": "desc/class_description_cls7.txt"
        },
        "indoor_office": {
            "src_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/txt/indoor_office.box.txt",
            "src_map_path": "desc/class_description.indoor_office.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/cls7/indoor_office.map.box.txt",
            "dst_map_path": "desc/class_description_cls7.txt"
        },
        "ges30_cloudwalk_digit": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_cloudwalk.box.txt",
            "src_map_path": "desc/class_description.ges30_cloudwalk.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/digit/ges30_cloudwalk_digit.map.box.txt",
            "dst_map_path": "desc/class_description_digit.txt"
        },
        "ges30_imgs_1015_1017_digit": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_imgs_1015_1017.box.txt",
            "src_map_path": "desc/class_description.ges30_imgs_1015_1017.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/digit/ges30_imgs_1015_1017_digit.map.box.txt",
            "dst_map_path": "desc/class_description_digit.txt"
        },
        "ges30_imgs_1018_1019_digit": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_imgs_1018_1019.box.txt",
            "src_map_path": "desc/class_description.ges30_imgs_1018_1019.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/digit/ges30_imgs_1018_1019_digit.map.box.txt",
            "dst_map_path": "desc/class_description_digit.txt"
        },
        "indoor_multi_scenario_digit": {
            "src_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/txt/indoor_multi_scenario.box.txt",
            "src_map_path": "desc/class_description.indoor_multi_scenario.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/digit/indoor_multi_scenario_digit.map.box.txt",
            "dst_map_path": "desc/class_description_digit.txt"
        },
        "indoor_office_digit": {
            "src_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/txt/indoor_office.box.txt",
            "src_map_path": "desc/class_description.indoor_office.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/digit/indoor_office_digit.map.box.txt",
            "dst_map_path": "desc/class_description_digit.txt"
        },
        "ges30_cloudwalk_cls13": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_cloudwalk.box.txt",
            "src_map_path": "desc/class_description.ges30_cloudwalk_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/ges30_cloudwalk_cls13.map.box.txt",
            "dst_map_path": "desc/class_description_cls13.txt"
        },
        "ges30_imgs_1015_1017_cls13": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_imgs_1015_1017.box.txt",
            "src_map_path": "desc/class_description.ges30_imgs_1015_1017_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/ges30_imgs_1015_1017_cls13.map.box.txt",
            "dst_map_path": "desc/class_description_cls13.txt"
        },
        "ges30_imgs_1018_1019_cls13": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/txtv2/ges30_imgs_1018_1019.box.txt",
            "src_map_path": "desc/class_description.ges30_imgs_1018_1019_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/ges30_imgs_1018_1019_cls13.map.box.txt",
            "dst_map_path": "desc/class_description_cls13.txt"
        },
        "indoor_multi_scenario_cls13": {
            "src_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/txt/indoor_multi_scenario.box.txt",
            "src_map_path": "desc/class_description.indoor_multi_scenario_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/cls13/indoor_multi_scenario_cls13.map.box.txt",
            "dst_map_path": "desc/class_description_cls13.txt"
        },
        "indoor_office_cls13": {
            "src_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/txt/indoor_office.box.txt",
            "src_map_path": "desc/class_description.indoor_office_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/cls13/indoor_office_cls13.map.box.txt",
            "dst_map_path": "desc/class_description_cls13.txt"
        },
        "zptest_cls13": {
            "src_path": "/dataset/dataset/ssd/gesture/zptest_cls13/cls13/zptest_cls13.box.txt",
            "src_map_path": "/dataset/dataset/ssd/gesture/zptest_cls13/labels/class_description.zptest_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/zptest_cls13/cls13/zptest_cls13.map.box.txt",
            "dst_map_path": "desc/class_description_cls13.txt"
        },
        "ges30_cloudwalk_clseasy": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/crop/cls13/train.ges30_cloudwalk_cls13.map.box.match.txt",
            "src_map_path": "desc/class_description_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/clseasy/train.ges30_cloudwalk_clseasy.map.box.match.txt",
            "dst_map_path": "desc/class_description_clseasy.txt"
        },
        "ges30_imgs_1015_1017_clseasy": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/crop/cls13/train.ges30_imgs_1015_1017_cls13.map.box.match.txt",
            "src_map_path": "desc/class_description_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/clseasy/train.ges30_imgs_1015_1017_clseasy.map.box.match.txt",
            "dst_map_path": "desc/class_description_clseasy.txt"
        },
        "ges30_imgs_1018_1019_clseasy": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/crop/cls13/train.ges30_imgs_1018_1019_cls13.map.box.match.txt",
            "src_map_path": "desc/class_description_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/clseasy/train.ges30_imgs_1018_1019_clseasy.map.box.match.txt",
            "dst_map_path": "desc/class_description_clseasy.txt"
        },
        "indoor_multi_scenario_clseasy": {
            "src_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/crop/cls13/train.indoor_multi_scenario_cls13.map.box.match.txt",
            "src_map_path": "desc/class_description_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/clseasy/train.indoor_multi_scenario_clseasy.map.box.match.txt",
            "dst_map_path": "desc/class_description_clseasy.txt"
        },
        "indoor_office_clseasy": {
            "src_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/crop/cls13/train.indoor_office_cls13.map.box.match.txt",
            "src_map_path": "desc/class_description_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/clseasy/train.indoor_office_clseasy.map.box.match.txt",
            "dst_map_path": "desc/class_description_clseasy.txt"
        },
        "test_ges30_imgs_1018_1019_clseasy": {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/crop/cls13/test.ges30_imgs_1018_1019_cls13.map.box.match.txt",
            "src_map_path": "desc/class_description_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/clseasy/test.ges30_imgs_1018_1019_clseasy.map.box.match.txt",
            "dst_map_path": "desc/class_description_clseasy.txt"
        },
        "test_indoor_multi_scenario_clseasy": {
            "src_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/crop/cls13/test.indoor_multi_scenario_cls13.map.box.txt",
            "src_map_path": "desc/class_description_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/clseasy/test.indoor_multi_scenario_clseasy.map.box.txt",
            "dst_map_path": "desc/class_description_clseasy.txt"
        },
        "zptest_clseasy": {
            "src_path": "/dataset/dataset/ssd/gesture/zptest_cls13/crop/cls13/zptest_cls13.map.box.txt",
            "src_map_path": "desc/class_description_cls13.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/zptest_cls13/clseasy/zptest_cls13.map.box.txt",
            "dst_map_path": "desc/class_description_clseasy.txt"
        },
        "ges30_cloudwalk_" + cls_name: {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/crop/txtv2/ges30_cloudwalk.box.match.txt",
            "src_map_path": "desc/class_description.ges30_cloudwalk_cls14.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/ges30_cloudwalk.map.box.match.txt",
            "dst_map_path": "desc/class_description_" + cls_name + ".txt"
        },
        "ges30_imgs_1015_1017_" + cls_name: {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/crop/txtv2/ges30_imgs_1015_1017.box.match.txt",
            "src_map_path": "desc/class_description.ges30_imgs_1015_1017_cls14.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/ges30_imgs_1015_1017.map.box.match.txt",
            "dst_map_path": "desc/class_description_" + cls_name + ".txt"
        },
        "ges30_imgs_1018_1019_" + cls_name: {
            "src_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/crop/txtv2/ges30_imgs_1018_1019.box.match.txt",
            "src_map_path": "desc/class_description.ges30_imgs_1018_1019_cls14.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/ges30_imgs_1018_1019.map.box.match.txt",
            "dst_map_path": "desc/class_description_" + cls_name + ".txt"
        },
        "indoor_multi_scenario_" + cls_name: {
            "src_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/crop/txt/indoor_multi_scenario.box.match.txt",
            "src_map_path": "desc/class_description.indoor_multi_scenario_cls14.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/" + cls_name + "/indoor_multi_scenario.map.box.match.txt",
            "dst_map_path": "desc/class_description_" + cls_name + ".txt"
        },
        "indoor_office_" + cls_name: {
            "src_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/crop/txt/indoor_office.box.match.txt",
            "src_map_path": "desc/class_description.indoor_office_cls14.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/indoor_office_mobile_0412/" + cls_name + "/indoor_office.map.box.match.txt",
            "dst_map_path": "desc/class_description_" + cls_name + ".txt"
        },
        "zptest_" + cls_name: {
            "src_path": "/dataset/dataset/ssd/gesture/zptest/crop/txt/zptest.box.match.txt",
            "src_map_path": "/dataset/dataset/ssd/gesture/zptest/txt/class_description.zptest.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/zptest/" + cls_name + "/zptest.map.box.match.txt",
            "dst_map_path": "desc/class_description_" + cls_name + ".txt"
        },
        "jsc220713_ai": {
            "src_path": "/dataset/dataset/ssd/gesture/jiashicang/txt/jsc220713.box.match.txt",
            "src_map_path": "/dataset/dataset/ssd/gesture/jiashicang/txt/class_description.jsc220713.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/jiashicang/ai/txt/jsc220713.map.box.match.txt",
            "dst_map_path": "desc/class_description_" + cls_name + ".txt"
        },

        "hagrid_" + cls_name: {
            "src_path": "/dataset/dataset/ssd/gesture/hagrid/txt/train.hagrid.txt",
            "src_map_path": "/dataset/dataset/ssd/gesture/hagrid/txt/class_description.hagrid.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/hagrid/" + cls_name + "/train.hagrid.map.box.txt",
            "dst_map_path": "desc/class_description_" + cls_name + ".txt"
        },

        "jsc220713_" + cls_name: {
            "src_path": "/dataset/dataset/ssd/gesture/jiashicang/resize/txtv2/test.jsc220713.gt.txt",
            "src_map_path": "/dataset/dataset/ssd/gesture/jiashicang/resize/txt/class_description.jsc220713.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/test.jsc220713.gt.txt",
            "dst_map_path": "desc/class_description_" + cls_name + ".txt"
        },

        "jsc220718_day_" + cls_name: {
            "src_path": "/dataset/dataset/ssd/gesture/jiashicang/resize/txtv2/jsc220718_day.txt",
            "src_map_path": "/dataset/dataset/ssd/gesture/jiashicang/jsc220718_day/txt/class_description.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/jsc220718_day.map.box.txt",
            "dst_map_path": "desc/class_description_" + cls_name + ".txt"
        },

        "jsc220715_garage_" + cls_name: {
            "src_path": "/dataset/dataset/ssd/gesture/jiashicang/resize/txtv2/jsc220715_garage.txt",
            "src_map_path": "/dataset/dataset/ssd/gesture/jiashicang/jsc220715_garage/txt/class_description.txt",
            "dst_path": "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/jsc220715_garage.map.box.txt",
            "dst_map_path": "desc/class_description_" + cls_name + ".txt"
        },
    }
    # datanames = ["ges30_cloudwalk", "ges30_imgs_1015_1017", "ges30_imgs_1018_1019", "indoor_multi_scenario", "indoor_office"]
    # datanames = ["ges30_cloudwalk_digit", "ges30_imgs_1015_1017_digit", "ges30_imgs_1018_1019_digit", "indoor_multi_scenario_digit", "indoor_office_digit"]
    datanames = ["ges30_cloudwalk_" + cls_name, "ges30_imgs_1015_1017_" + cls_name, "ges30_imgs_1018_1019_" + cls_name,
                 "indoor_multi_scenario_" + cls_name, "indoor_office_" + cls_name, "zptest_" + cls_name,
                 "jsc220713_" + cls_name, "jsc220718_day_" + cls_name, "jsc220715_garage_" + cls_name]
    datanames = ["jsc220715_garage_" + cls_name]

    for dataname in datanames:
        print("deal {}".format(dataname))
        src_path = datas[dataname]["src_path"]
        src_map_path = datas[dataname]["src_map_path"]
        dst_path = datas[dataname]["dst_path"]
        dst_map_path = datas[dataname]["dst_map_path"]
        print("src_path: {}".format(src_path))
        print("dst_path: {}".format(dst_path))
        get_map_txt(src_map_path, dst_map_path, src_path, dst_path)
