import os, sys
import numpy as np
import cv2
from progressbar import Timer, Bar, ETA, Percentage, ProgressBar
import collections
import shutil
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.box_and_kpt_utils import update_box, update_kpt, get_crop_size, get_resize_img
from utils.get_split_set import get_split_set
from utils.get_path_len import get_path_len

kpt_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (
    255, 0, 255)]  # left_eye(green), right_eye(red), nose(blue), left_mouth(cyan), right_mouth(yellow), center(magenta)
PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
           (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
           (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
           (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
           (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
           (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
           (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
           (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
           (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
           (134, 134, 103), (145, 148, 174), (255, 208, 186),
           (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
           (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
           (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
           (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
           (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
           (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
           (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
           (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
           (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
           (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
           (191, 162, 208)]


def draw_box_kpt_from_cwface_path(list_path, save_img_root, save_txt_path):
    img_path_list = []
    labels = []
    box_list = []
    kpts_list = []
    box_pos = 2
    kpt_pos = 6
    with open(list_path, "r") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path_list.append(line_sp[0])
            labels.append(line_sp[1])
            box = list(map(float, line_sp[box_pos:box_pos + 4]))  # 通过tools/cleanse/demo.py检测框x, y, w, h(左上角宽高)
            # box = [box[1], box[0], box[1] + box[3], box[0] + box[2]]
            box_list.append(box)
            kpt = list(map(float, line_sp[kpt_pos:kpt_pos + 10]))  # 通过tools/cleanse/demo.py提取的原图的关键点存储形式x,y
            # kpt = np.array(kpt)
            # kpt = kpt.reshape(-1, 2)
            # kpt = kpt[:, [1, 0]]
            # kpt = kpt.reshape(-1)
            kpts_list.append(kpt)  # total 5 points: left_eye, right_eye, nose, left_mouth, right_mouth

    l = len(img_path_list)
    widgets = ['draw_bos_kpt: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets)
    with open(save_txt_path, "w") as fw:
        for index, img_path in enumerate(pbar(img_path_list)):
            im = draw_box_kpt_with_cwface_label_over_image(img_path, [box_list[index]], [kpts_list[index]])
            save_path = os.path.join(save_img_root, os.path.basename(img_path))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, im)
            im = cv2.imread(save_path)
            cnt = 0
            while im is None and cnt < 10:
                print("Error save {}, cnt: {}".format(save_path, cnt + 1))
                cv2.imwrite(save_path, im)
                im = cv2.imread(save_path)
                cnt += 1
            s = save_path + " " + labels[index] + " 1" + "\n"
            fw.write(s)

    return


def draw_box_kpt_from_cwface_path_same_image(list_path, save_img_root, save_txt_path, src_img_root=None, radius=3,
                                             box_thick=3, is_crop_view=False, view_index=False):
    labels = collections.defaultdict()
    box_dict = collections.defaultdict(list)
    kpts_dict = collections.defaultdict(list)
    box_pos = 2
    kpt_pos = 6
    with open(list_path, "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line_sp = line.strip().split(" ")
            img_path = line_sp[0]
            labels[img_path] = line_sp[1]
            box = list(map(float, line_sp[box_pos:box_pos + 4]))  # 通过tools/cleanse/demo.py检测框x, y, w, h(左上角宽高)
            # box = [box[1], box[0], box[1] + box[3], box[0] + box[2]]
            box_dict[img_path].append(box)
            kpt = list(map(float, line_sp[kpt_pos:kpt_pos + 10]))  # 通过tools/cleanse/demo.py提取的原图的关键点存储形式x,y
            # kpt = np.array(kpt)
            # kpt = kpt.reshape(-1, 2)
            # kpt = kpt[:, [1, 0]]
            # kpt = kpt.reshape(-1)
            kpts_dict[img_path].append(kpt)  # total 5 points: left_eye, right_eye, nose, left_mouth, right_mouth

    l = len(box_dict)
    widgets = ['draw_bos_kpt: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets)
    os.makedirs(os.path.dirname(save_txt_path).encode("utf-8"), exist_ok=True)
    with open(save_txt_path, "w", encoding="utf-8") as fw:
        for index, img_path in enumerate(pbar(box_dict)):
            im = draw_box_kpt_with_cwface_label_over_image(img_path, box_dict[img_path], kpts_dict[img_path],
                                                           radius=radius, box_thick=box_thick,
                                                           is_crop_view=is_crop_view, view_index=view_index)
            if src_img_root is not None:
                if src_img_root.endswith("/"):
                    l = len(src_img_root)
                else:
                    l = len(src_img_root) + 1
                img_name = img_path[l:]
            else:
                img_name = img_path.strip("/")
            if img_name.endswith(".ppm"):
                img_name = img_name[:-4] + ".jpg"
            if ".jpg." in img_name:
                img_name = ".".join(img_name.split(".")[:-1])
            save_path = os.path.join(save_img_root, img_name)
            os.makedirs(os.path.dirname(save_path).encode("utf-8"), exist_ok=True)
            cv2.imwrite(save_path, im)
            im = cv2.imread(save_path)
            cnt = 0
            while im is None and cnt < 10:
                print("Error save {}, cnt: {}".format(save_path, cnt + 1))
                cv2.imwrite(save_path, im)
                im = cv2.imread(save_path)
                cnt += 1
            s = save_path + " " + labels[img_path] + " 1" + "\n"
            fw.write(s)

    return


def draw_box_kpt_from_tang_path(src_txt_path, save_img_root, save_txt_path):
    cnt = 0
    with open(src_txt_path, "r") as fr:
        with open(save_txt_path, "w") as fw:
            for line in fr.readlines():
                cnt += 1
                # if cnt > 10:
                #     break
                line_sp = line.strip().split(" ")
                img_path = line_sp[0]
                box = list(map(float, line_sp[3:7]))  # y0, x0, y1, x1
                kpt = list(map(float, line_sp[
                                      7:17]))  # y, x total 5 points: left_eye, right_eye, nose, left_mouth, right_mouth
                im = draw_box_kpt_with_tang_label(img_path, [box], [kpt])
                save_path = os.path.join(save_img_root, os.path.basename(img_path))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, im)
                s = save_path + " -1" + " 1" + "\n"
                fw.write(s)
    return


def draw_box_kpt_with_tang_label(im, box_list, kpts_list, normalized=False, thick=3, dst_size=(128, 128),
                                 draw_box=True):
    if isinstance(im, str):
        im = cv2.imread(im)
    elif isinstance(im, np.ndarray):
        im = im
    height = im.shape[0]
    width = im.shape[1]

    if height == 0 or width == 0:
        return None

    font_width = 10
    kpts_arr = np.array(kpts_list)
    kpts_arr = np.reshape(kpts_arr, (-1, 5, 2))

    for box in box_list:
        box = np.array(box)  # y0, x0, y1, x1
        if normalized:
            box[1] = box[1] * width
            box[3] = box[3] * width
            box[0] = box[0] * height
            box[2] = box[2] * height
            kpts_arr = kpts_arr * np.array([[[dst_size[1], dst_size[0]]]], dtype=np.float32)
        box = box.astype(np.int32)

        if draw_box:
            cv2.rectangle(im, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 3)

    if normalized is not True:
        kpts_arr = kpts_arr / np.array([[height, width]], dtype=np.float32) * np.array([dst_size[0], dst_size[1]],
                                                                                       dtype=np.float32)

    im = cv2.resize(im, dst_size)

    for kpts in kpts_arr:
        kpts = kpts.astype(np.int32)  # y, x total 5 points: left_eye, right_eye, nose, left_mouth, right_mouth
        for kpt_idx, kpt in enumerate(kpts):
            cv2.circle(im, (kpt[1], kpt[0]), thick, kpt_color[kpt_idx], -1)

    return im


def draw_box_kpt_with_cwface_label(im, box_list, kpts_list, normalized=False, radius=3, dst_size=(128, 128),
                                   draw_box=True, box_thick=3):
    if isinstance(im, str):
        im = cv2.imread(im)
    elif isinstance(im, np.ndarray):
        im = im
    height = im.shape[0]
    width = im.shape[1]
    dst_h, dst_w = dst_size

    if height == 0 or width == 0:
        return None

    font_width = 10
    im = cv2.resize(im, (dst_w, dst_h))
    for box in box_list:
        box = np.array(box)  # x, y, w, h
        if normalized:
            box[0] = box[0] * width
            box[1] = box[1] * height
            box[2] = box[2] * width
            box[3] = box[3] * height

        if normalized is not True:
            box = box / np.array([width, height, width, height], dtype=np.float32) * \
                  np.array([dst_w, dst_h, dst_w, dst_h], dtype=np.float32)

        if draw_box:
            box = box.astype(np.int32)
            cv2.rectangle(im, (box[0], box[1]), (box[2] + box[0], box[3] + box[1]), (255, 0, 0), thickness=box_thick)

    kpts_arr = np.array(kpts_list)
    kpts_arr = np.reshape(kpts_arr, (-1, 5, 2))
    kpts_arr = kpts_arr / np.array([[width, height]], dtype=np.float32) * np.array([dst_w, dst_h], dtype=np.float32)
    for kpts in kpts_arr:
        kpts = kpts.astype(np.int32)  # x, y total 5 points: left_eye, right_eye, nose, left_mouth, right_mouth
        for kpt_idx, kpt in enumerate(kpts):
            cv2.circle(im, (kpt[0], kpt[1]), radius=radius, color=kpt_color[kpt_idx], thickness=-1)

    return im


def draw_box_kpt_with_cwface_label_over_image(im, box_list, kpts_list=None, normalized=False, radius=3,
                                              dst_size=(416, 416), draw_box=True, kpt_num=5, box_thick=3,
                                              is_crop_view=False, view_index=False, resize=False, box_color=(255, 0, 0),
                                              box_txt=None, box_fontScale=0.3, kpt_fontScale=0.3):
    if isinstance(im, str):
        im = cv2.imread(im)
    elif isinstance(im, np.ndarray):
        im = im
    if kpts_list is not None:
        kpts_list = np.array(kpts_list).reshape((-1, kpt_num, 2))
    height = im.shape[0]
    width = im.shape[1]
    dst_h, dst_w = dst_size
    box_list = np.array(box_list).reshape(-1, 4)
    box_num = box_list.shape[0]
    if isinstance(box_color, tuple) and len(box_color) == 3:
        box_color = [box_color]
    if box_txt is not  None and isinstance(box_txt, str):
        box_txt = [box_txt]
    if box_num >= 2:
        if box_txt is not None:
            assert box_num == len(box_txt), "the box len({})  not equal box_txt len({})".format(box_num, len(box_txt))
            if box_num != len(box_color):
                new_box_color = []
                for i in range(1, box_num):
                    new_box_color.append(box_color[0])
                box_color = new_box_color

    if is_crop_view:  # 由于有些原图太大不方便显示因此对其进行裁剪
        box_np = np.array(box_list)
        box_np[:, 2] = box_np[:, 0] + box_np[:, 2]  # 因为box的存储是按照x,y,w,h存储的，这里先变换成坐标
        box_np[:, 3] = box_np[:, 1] + box_np[:, 3]
        x_min = max(np.min(box_np[:, 0]), 0)
        x_max = min(np.max(box_np[:, 2]), width)
        y_min = max(np.min(box_np[:, 1]), 0)
        y_max = min(np.max(box_np[:, 3]), height)
        box = [x_min, y_min, x_max - x_min, y_max - y_min]
        crop_x1, crop_y1, crop_x2, crop_y2 = get_crop_size(box, im.shape)
        crop_bbox = []
        for bbox in box_list:
            crop_bbox.append(update_box(bbox, crop_x1, crop_y1))
        box_list = crop_bbox

        if kpts_list is not None:
            crop_kpts = []
            for kpt in kpts_list:
                crop_kpts.append(update_kpt(kpt, crop_x1, crop_y1))
            kpts_list = np.array(crop_kpts).reshape((-1, kpt_num, 2))
        im = im[crop_y1:crop_y2, crop_x1:crop_x2, :]
        height = im.shape[0]
        width = im.shape[1]
    if height == 0 or width == 0:
        return None
    if kpts_list is not None:
        kpts = np.array(kpts_list).reshape((-1, kpt_num, 2))
        crop_x1 = int(min(np.min(kpts[:, :, 0]), 0))  # 可能小于0
        crop_y1 = int(min(np.min(kpts[:, :, 1]), 0))  # 可能小于0
        crop_x2 = int(max(np.max(kpts[:, :, 0]), width))  # 可能大于width
        crop_y2 = int(max(np.max(kpts[:, :, 1]), height))  # 可能大于height
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        # ratio = max(crop_w / dst_w, crop_h / dst_h)
        # new_bbox_w = int(dst_w * ratio)
        # new_bbox_h = int(dst_h * ratio)
        new_bbox_w = crop_w
        new_bbox_h = crop_h
        crop_img = np.zeros((new_bbox_h, new_bbox_w, 3), dtype=im.dtype)
        crop_img[abs(crop_y1):abs(crop_y1) + height, abs(crop_x1):abs(crop_x1) + width] = im
        im = crop_img.copy()
        crop_bbox = []
        for bbox in box_list:
            crop_bbox.append(update_box(bbox, crop_x1, crop_y1))
        box_list = crop_bbox
        crop_kpts = []
        for kpt in kpts:
            crop_kpts.append(update_kpt(kpt, crop_x1, crop_y1))
        kpts_list = np.array(crop_kpts).reshape((-1, kpt_num, 2))

    font_width = 10
    im_h, im_w = im.shape[:2]
    if resize:
        im, box_list, kpt = get_resize_img(im, box_list, dst_h, dst_w, keep_ratio=True, kpt=kpts_list)
        im_h, im_w = im.shape[:2]

    for box_idx, box in enumerate(box_list):
        box = np.array(box)  # x, y, w, h
        if normalized:
            box[0] = box[0] * width
            box[1] = box[1] * height
            box[2] = box[2] * width
            box[3] = box[3] * height

        if draw_box:
            box = box.astype(np.int32)
            box_thick = int(max(min(im_w, im_h) * 0.006, 2))
            color = box_color[box_idx]
            cv2.rectangle(im, (box[0], box[1]), (box[2] + box[0], box[3] + box[1]), color, thickness=box_thick)
            if box_txt is not None:
                # text_size, baseline = cv2.getTextSize(str(box_txt), cv2.FONT_HERSHEY_SIMPLEX, box_fontScale, box_thick)
                box_fontScale = int(max(min(im_w, im_h) * 0.006, 2))
                txt = str(box_txt[box_idx])

                cv2.putText(im, txt, (box[0], max(int(box_fontScale + 1), box[1])),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=box_fontScale, color=color,
                            thickness=box_thick)
    if kpts_list is not None:
        kpts_arr = np.array(kpts_list)
        kpts_arr = np.reshape(kpts_arr, (-1, kpt_num, 2))
        if resize:
            kpts_arr = kpts_arr / np.array([[im_w, im_h]], dtype=np.float32) * np.array([dst_w, dst_h],
                                                                                        dtype=np.float32)
        for index, kpts in enumerate(kpts_arr):
            kpts = kpts.astype(np.int32)  # x, y total 5 points: left_eye, right_eye, nose, left_mouth, right_mouth
            for kpt_idx, kpt in enumerate(kpts):
                cv2.circle(im, (kpt[0], kpt[1]), radius=radius, color=kpt_color[kpt_idx], thickness=-1)
                if view_index:
                    cv2.putText(im, str(index), (kpt[0] - 2, max(0, kpt[1] - 5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=kpt_fontScale, color=(0, 0, 0), thickness=1)

    return im


def draw_box_and_kpt(im, box, kpts=None, kpt_num=5, box_color=(255, 0, 0), box_txt=None, box_fontScale=10, box_thick=5,
                     is_crop_view=False, resize=False, dst_size=(416, 416)):
    """

    Args:
        im: ndarray or str
        box: [[x,y,w,h], ...]
        kpts: [x1,y1,...,xn,yn]
        box_txt:

    Returns:

    """
    im = draw_box_kpt_with_cwface_label_over_image(im, box_list=box, kpts_list=kpts, kpt_num=kpt_num,
                                                   box_color=box_color, box_txt=box_txt, box_fontScale=box_fontScale,
                                                   box_thick=box_thick, is_crop_view=is_crop_view, resize=resize,
                                                   dst_size=dst_size)
    return im


def draw_main_worker(compare_info_dict, split_set, img_path_list,
                     true_boxes, true_kpts,
                     pred_boxes, pred_kpts,
                     thread_idx, dst_dir, indices=None,
                     baseline_bboxes=None, baseline_kpts=None, src_img_root=None,
                     dst_path=None, values=None, values_name=None, ):
    print("thread_idx:{} split_index: {}".format(thread_idx, split_set[thread_idx]))
    current_set = split_set[thread_idx]
    if dst_path is not None:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        key_num = values.shape[1]
        assert key_num == len(values_name), "values_name {} not equal {}".format(values_name, key_num)
        fw = open(dst_path, "w")
    for idx in range(current_set[0], current_set[1]):
        if indices is None:
            current_indice = idx
        else:
            current_indice = indices[idx]

        img_path = img_path_list[current_indice]
        if src_img_root is not None:
            if src_img_root.endswith("/"):
                l = len(src_img_root)
            else:
                l = len(src_img_root) + 1
            img_name = img_path[l:]
        else:
            img_name = os.path.basename(img_path)
        if "truncv2" in img_path and not img_name.startswith("RetainRightMouthAndNose_"):
            continue
        src_im = cv2.imread(img_path)

        box = true_boxes[current_indice]
        true_kpt = true_kpts[current_indice]
        crop_x1, crop_y1, crop_x2, crop_y2 = get_crop_size(box, src_im.shape)
        src_im = src_im[crop_y1:crop_y2, crop_x1:crop_x2, :]
        box = update_box(box, crop_x1, crop_y1)
        true_kpt = update_kpt(true_kpt, crop_x1, crop_y1)
        true_im = draw_box_and_kpt(src_im.copy(), box, true_kpt)

        im_h = true_im.shape[0]
        blank = np.ones([im_h, 10, 3], dtype=np.uint8) * 128
        if baseline_bboxes is not None:
            baseline_box = baseline_bboxes[current_indice]
            baseline_kpt = baseline_kpts[current_indice]
            baseline_box = update_box(baseline_box, crop_x1, crop_y1)
            baseline_kpt = update_kpt(baseline_kpt, crop_x1, crop_y1)
            baseline_im = draw_box_and_kpt(src_im.copy(), baseline_box, baseline_kpt)

        pred_box = pred_boxes[current_indice]
        pred_kpt = pred_kpts[current_indice]
        pred_box = update_box(pred_box, crop_x1, crop_y1)
        pred_kpt = update_kpt(pred_kpt, crop_x1, crop_y1)
        pred_im = draw_box_and_kpt(src_im.copy(), pred_box, pred_kpt)

        if img_name.endswith(".ppm"):
            img_name = img_name[:-4] + ".jpg"
        if ".jpg." in img_name:
            img_name = ".".join(img_name.split(".")[:-1])
        save_path = os.path.join(dst_dir, "{:0>5d}_{}".format(idx, img_name))
        if baseline_bboxes is not None:
            im = np.concatenate([pred_im, blank, true_im, blank, baseline_im], axis=1)
        else:
            im = np.concatenate([pred_im, blank, true_im], axis=1)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, im)
        if dst_path is not None:
            val = values[idx].reshape(-1).tolist()
            val_str = ""
            for i, k in enumerate(values_name):
                if i == 0:
                    val_str += k + ":" + "{:.2f}".format(val[i])
                else:
                    val_str += " " + k + ":" + "{:.2f}".format(val[i])
            fw.write(save_path + " 0 1 " + val_str + "\n")
    if dst_path is not None:
        fw.close()


def mp_draw_box_kpts(compare_info_dict, img_path_list,
                     true_boxes, true_kpts,
                     pred_boxes, pred_kpts,
                     dst_dir, indices=None,
                     baseline_bboxes=None, baseline_kpts=None, src_img_root=None,
                     dst_path=None, values=None, values_name=None,
                     thread_num=1):
    draw_num = len(img_path_list)
    split_set = get_split_set(total_num=draw_num, thread_num=thread_num)
    thread_list = []

    if os.path.isdir(dst_dir) is not True:
        os.mkdir(dst_dir)
    draw_main_worker(compare_info_dict, split_set, img_path_list,
                     true_boxes, true_kpts,
                     pred_boxes, pred_kpts,
                     0, dst_dir, indices,
                     baseline_bboxes, baseline_kpts, src_img_root,
                     dst_path, values, values_name)
    # for thread_idx in range(thread_num):
    #     p = mp.Process(target=draw_main_worker, args=(
    #         split_set, img_path_list, true_boxes, pred_boxes, baseline_bboxes, true_kpts, pred_kpts, baseline_kpts,
    #         indices, thread_idx, dst_dir))
    #     p.start()
    #     thread_list.append(p)
    #
    # for p in thread_list:
    #     p.join()


def save_img(save_path, img):
    os.makedirs(os.path.dirname(save_path).encode("utf-8"), exist_ok=True)
    cv2.imwrite(save_path, img)


def show_img(img):
    plt.imshow(img[:, :, ::-1])
    plt.show()


def save_img_with_box(img_path, src_img_prefix, save_img_root, bbox, box_txt, box_color_list, box_num=None,
                      is_crop_view=False, show=False, resize=False, dst_size=(416, 416)):
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    img = draw_box_and_kpt(img, bbox, box_color=box_color_list, box_txt=box_txt, is_crop_view=is_crop_view,
                           resize=resize, dst_size=dst_size)
    if show:
        show_img(img)
    img_path_sp = img_path[get_path_len(src_img_prefix):].split(".")
    if box_num is not None:
        img_path_sp[-2] = img_path_sp[-2] + "box" + str(box_num)
    img_name = ".".join(img_path_sp)
    new_save_img_path = os.path.join(save_img_root, img_name)
    save_img(new_save_img_path, img)
    return new_save_img_path


if __name__ == '__main__':
    # list_path = "/dataset/dataset/ssd/kpt/cluster_rst/sy.ssd.list"
    # save_img_root = "/ssd/128x128/tang.v8.st/shenyang.orig/mask_dy_draw"
    # save_txt_path = "/dataset/dataset/ssd/kpt/cluster_rst/sy.ssd.draw.list"
    # list_path = "/dataset/dataset/ssd/kpt/cluster_rst/sy.ssd.tiny.list"
    # save_img_root = "/dataset/dataset/ssd/kpt/imgs"
    # save_txt_path = "/dataset/dataset/ssd/kpt/cluster_rst/sy.ssd.tiny.draw.list"
    # save_img_root = "/dataset/dataset/ssd/kpt/kpt_test_img"
    # list_path = "/dataset/dataset/ssd/kpt/cluster_rst/det/det.lst"
    # save_img_root = "/dataset/dataset/ssd/kpt/kpt_test_img"
    # save_txt_path = "/dataset/dataset/ssd/kpt/cluster_rst/kpt_test_img.draw.list"
    # os.makedirs(save_img_root, exist_ok=True)

    # draw_bos_kpt_from_list(list_path, save_img_root, save_txt_path)

    # list_path = "/dataset/dataset/ssd/kpt_train_list/JD/gt.cascade.cwface.txt"
    # save_img_root = "/dataset/dataset/ssd/kpt_data/draw/JD/cascade"
    # save_txt_path = "/dataset/dataset/ssd/kpt_train_list/JD/draw/cascade.list"
    # save_txt_path2 = "/dataset/dataset/ssd/kpt/cluster_rst/JD.draw.cascade.list"
    # os.makedirs(save_img_root, exist_ok=True)
    # os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
    # os.makedirs(os.path.dirname(save_txt_path2), exist_ok=True)
    # # draw_box_kpt_from_tang_path(list_path, save_img_root, save_txt_path)
    # draw_box_kpt_from_cwface_path(list_path, save_img_root, save_txt_path)
    # shutil.copyfile(save_txt_path, save_txt_path2)
    print("visualize")
    list_path = "/ssd/test_db.orig/paired/DY5K.TINY/xc.tiny400rs100m2.list"
    root = "/ssd/test_db.orig/paired"
    region = "DY5K.TINY"
    gt_det_name = "xc.tiny400rs100m2.list"

    save_img_root = "/dataset/dataset/ssd/kpt_data/draw/" + region + "/imgs/" + gt_det_name
    save_txt_path = os.path.join(root, region, "draw", gt_det_name)
    print("save draw txt {}".format(save_txt_path))
    # if os.path.exists(save_img_root):
    #     shutil.rmtree(save_img_root)
    os.makedirs(save_img_root, exist_ok=True)
    src_img_root = None
    draw_box_kpt_from_cwface_path_same_image(list_path, save_img_root, save_txt_path, src_img_root=src_img_root,
                                             radius=3, box_thick=3, is_crop_view=True, view_index=False)
    save_txt_path2 = "/dataset/dataset/ssd/kpt/cluster_rst/" + region + ".draw." + gt_det_name
    print("save go label txt {}".format(save_txt_path2))
    shutil.copyfile(save_txt_path, save_txt_path2)
