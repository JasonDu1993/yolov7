# -*- coding: utf-8 -*-
# @Time    : 2021/9/29 14:45
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : gen_golabel.py
# @Software: PyCharm
import os
import cv2

from utils.draw_box_kpt_utils import save_img_with_box
from utils.gen_txt import get_path_len
from utils.datas.gesture.class_color_c9 import CLASSES, PALETTE


def traverse(src_dir, exts=None):
    if exts is None:
        exts = ['.jpg', '.png', '.bmp']
    labels = {}
    result = []
    cnt = -1
    for root, dirs, files in sorted(os.walk(src_dir)):
        if len(files) > 0:
            class_name = root[get_path_len(src_dir):]
            if class_name not in labels:
                cnt += 1
                labels[class_name] = str(cnt)
            for f in sorted(files):
                ext = os.path.splitext(f)[-1].lower()
                if ext in exts:
                    result.append([os.path.join(root, f), str(cnt)])
    return result


def gen_golabel_from_txt(input_path, save_path, vis=False, is_contain_imgwh=True, src_img_prefix=None,
                         save_img_root=None, use_gt_label=False, verbose=False, resize=False, dst_size=(416, 416),
                         multi_box_in_one_img=False):
    cnt = 0
    boxnum_per_img = {}
    if os.path.isdir(input_path):
        result = traverse(input_path)
        with open(save_path, "w", encoding="utf-8") as fw:
            for lines in result:
                img_path, class_name = lines
                color = "1"
                new_line = img_path + " " + class_name + " " + color + "\n"
                fw.write(new_line)
                fw.flush()
    else:
        if not multi_box_in_one_img:
            with open(input_path, "r", encoding="utf-8") as fr:
                with open(save_path, "w", encoding="utf-8") as fw:
                    for line in fr.readlines():
                        cnt += 1
                        # if cnt % 1000 > 10:
                        #     continue
                        line_sp = line.strip().split(" ")
                        line_sp.insert(2, "1")
                        img_path = line_sp[0]
                        if verbose:
                            print("img_path: {}".format(img_path))
                        if img_path in boxnum_per_img:
                            boxnum_per_img[img_path] = boxnum_per_img[img_path] + 1
                        else:
                            boxnum_per_img[img_path] = 1
                        # del  line_sp[3]
                        # del  line_sp[3]

                        if vis:
                            label = int(line_sp[1])
                            if use_gt_label:
                                gt_label = int(line_sp[-1])

                            if is_contain_imgwh:
                                bbox = list(map(float, line_sp[5:9]))
                                current_idx = 9
                            else:
                                bbox = list(map(float, line_sp[3:7]))
                                current_idx = 7
                            box_txt = "{}".format(CLASSES[label])
                            for i in range(current_idx, len(line_sp)):
                                box_txt += ":" + line_sp[i]
                            new_save_img_path = save_img_with_box(img_path, src_img_prefix, save_img_root, bbox,
                                                                  box_txt, PALETTE[label], boxnum_per_img[img_path],
                                                                  resize=resize, dst_size=dst_size)
                            line_sp[0] = new_save_img_path
                            if use_gt_label:
                                line_sp[1] = str(gt_label)
                            new_line = " ".join(line_sp[:3]) + "\n"
                            # print(new_line)
                            fw.write(new_line)
                            fw.flush()
                        else:
                            # print(line_sp)
                            fw.write(" ".join(line_sp) + "\n")
                            fw.flush()
        else:
            img_to_box = {}
            with open(input_path, "r", encoding="utf-8") as fr:
                for line in fr.readlines():
                    cnt += 1
                    # if cnt % 1000 > 10:
                    #     continue
                    line_sp = line.strip().split(" ")
                    img_path = line_sp[0]
                    if verbose:
                        print("img_path: {}".format(img_path))
                    if img_path in img_to_box:
                        img_to_box[img_path].append(line_sp)
                    else:
                        img_to_box[img_path] = [line_sp]
            with open(save_path, "w", encoding="utf-8") as fw:
                for img_path in img_to_box:
                    bbox_list = []
                    bbox_txt_list = []
                    bbox_color_list = []
                    box_num = len(img_to_box[img_path])
                    if box_num >= 2:
                        print("it have box num > 2, {}".format(img_path))
                    for line_sp in img_to_box[img_path]:
                        label = int(line_sp[1])
                        if is_contain_imgwh:
                            bbox = list(map(float, line_sp[4:8]))
                            current_idx = 9
                        else:
                            bbox = list(map(float, line_sp[2:6]))
                            current_idx = 7
                        bbox_list.append(bbox)
                        box_txt = "{}".format(CLASSES[label])
                        for i in range(current_idx, len(line_sp)):
                            box_txt += ":" + line_sp[i]
                        bbox_txt_list.append(box_txt)
                        bbox_color_list.append(PALETTE[int(label)])
                    new_save_img_path = save_img_with_box(img_path, src_img_prefix, save_img_root, bbox_list,
                                                          bbox_txt_list, bbox_color_list, resize=resize,
                                                          dst_size=dst_size)
                    new_line = new_save_img_path + " " + str(box_num) + " 1 \n"
                    fw.write(new_line)
                    fw.flush()


def get_docker_cmd(root_dir, port):
    port = str(port)
    name = "go" + port
    cmd = "sudo docker run --rm --name " + name + " -v " + root_dir + ":/zd " + \
          "-v /cloudgpfs/workspace/zhoudu/zhoudu:/zhoudu " + \
          "-v /cloudgpfs/workspace/zhoudu/zhoudu_data:/dataset " + \
          "-v /cloudgpfs/workspace/zhouyafei:/zhouyafei " + \
          "-p " + port + ":80 artifact.cloudwalk.work/rd_docker_dev/label-app/go 10.128.8.141:" + port + " /zd"
    print(cmd)
    cmd2 = "sudo docker rm -f " + name
    print(cmd2)


if __name__ == '__main__':
    cls_name = "c9"
    # src_dir = "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/train.ges30_cloudwalk.map.box.match.txt"
    # src_dir = "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/train.ges30_imgs_1015_1017.map.box.match.txt"
    # src_dir = "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/train.ges30_imgs_1018_1019.map.box.match.txt"
    # src_dir = "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/" + cls_name + "/train.indoor_multi_scenario.map.box.match.txt"
    # src_dir = "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/train.jsc220713.map.box.match.txt"
    src_dir = "/dataset/dataset/ssd/gesture/jiashicang/c9/test.jsc220718_day.map.box.txt"

    dataset = "test.jsc220718_day." + cls_name
    # traverse(src_dir)
    server_dir = "/cloudgpfs/workspace/zhoudu/zhoudu/golabel"
    docker_dir = "/zhoudu/golabel"
    # server_dir = "/cloudgpfs/workspace/zhoudu/zhoudu/test/gesture/show"
    # docker_dir = "/zhoudu/test/gesture/show"
    # task = "yolov3_416_v3"
    task = "demo"
    port = 50020
    multi_box_in_one_img = False
    # task = "gesture"
    # port = 50040
    root_dir = os.path.join(server_dir, task)
    save_path = os.path.join(docker_dir, task, "cluster_rst", dataset + ".txt")
    vis = True
    src_img_prefix = "/dataset/dataset/ssd/gesture/"
    save_img_root = os.path.join(docker_dir, task, "imgs")
    print("src path: {}".format(src_dir))
    print("save path: {}".format(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(save_img_root, exist_ok=True)
    resize = False

    gen_golabel_from_txt(src_dir, save_path, vis=vis, is_contain_imgwh=True, src_img_prefix=src_img_prefix,
                         save_img_root=save_img_root, resize=resize, multi_box_in_one_img=multi_box_in_one_img)
    get_docker_cmd(root_dir, port)
