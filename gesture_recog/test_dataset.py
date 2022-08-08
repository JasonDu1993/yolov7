# -*- coding: utf-8 -*-
# @Time    : 2022/7/7 17:28
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : extract_dataset.py
# @Software: PyCharm
import os
from gesture_recog.extract_box import extract_box
from gesture_recog.extract_gesture import extract_gesture
from mmdet.datasets.gesture.merge import get_merge_txt_seq
from tools.metric.eval_acc_and_roc import eval_acc_and_roc
from tools.metric.write_info import parse_error, save_false_det, save_result, save_miss_det
from tools.vis.gen_golabel import get_docker_cmd

if __name__ == '__main__':
    # yolov3
    # det_cfg = "configs/yolo_for_gesture/yolov3_416_relabel.py"
    # det_thd = 0.05
    # det_ckp = "/zhoudu/checkpoints/gesture/work_dirs/yolo_for_gesture/yolov3_416_relabel/epoch_50.pth"
    # use_cmd_for_det = False
    # yolov7
    det_thd = 0.1
    det_ckp = "/zhoudu/checkpoints/gesture/yolov7/yolov7_tiny_jsc/weights/deploy.best.pt"  # 一定要改det_thd的值
    use_cmd_for_det = True
    # 是否重新预测
    overwrite = True
    cls_name = "c9"
    model_name = "exp_c9_affine_hagrid_heart_jsc"
    recog_kwarg = {
        "config": "tools/gesture_recog/" + model_name + "/config.py",
        "model": "gesture_recog." + model_name + ".model.Model",
        "weight_path": "/zhoudu/checkpoints/gesture_recog/" + model_name + "/checkpoint/model-epoch19.weights",
        "first_sing_vec_path": "/zhoudu/test/gesture/feas/" + model_name + "/first_sing_vec.train.ges30_cloudwalk.ges30_imgs_1015_1017.npy"
    }

    datas = {
        "test_ges30_imgs_1018_1019_cls14": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/v2/test.ges30_imgs_1018_1019.map.box.txt",

        "test_ges30_imgs_1018_1019_cls7": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls7/test.ges30_imgs_1018_1019.map.box.txt",
        "test_indoor_multi_scenario_cls7": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/cls7/test.indoor_multi_scenario.map.box.txt",

        "test_neg_cw": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/neg_list/neg.test.ges30_imgs_1018_1019.map.box.txt",
        "test_neg_zipai": "/dataset/dataset/ssd/gesture/neg/txt/neg.map.box.txt",
        "test_neg_imagenet": "/dataset/dataset/ssd/gesture/imagenet/imagenet_ood.box.txt",

        "test_ges30_imgs_1018_1019_cls13": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/cls13/test.ges30_imgs_1018_1019_cls13.map.box.txt",
        "test_indoor_multi_scenario_cls13": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/cls13/test.indoor_multi_scenario_cls13.map.box.txt",

        "zptest_cls13": "/dataset/dataset/ssd/gesture/zptest_cls13/cls13/zptest_cls13.map.box.txt",

        "test_ges30_imgs_1018_1019_clseasy": "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/clseasy/test.ges30_imgs_1018_1019_clseasy.map.box.txt",
        "test_indoor_multi_scenario_clseasy": "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/clseasy/test.indoor_multi_scenario_clseasy.map.box.txt",
        "zptest_cls13_clseasy": "/dataset/dataset/ssd/gesture/zptest_cls13/clseasy/zptest_cls13.map.box.txt",

        "test_ges30_imgs_1018_1019_" + cls_name: "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/" + cls_name + "/test.ges30_imgs_1018_1019.map.box.match.txt",
        "test_indoor_multi_scenario_" + cls_name: "/dataset/dataset/ssd/gesture/indoor_multi_scenario_mobile_0326/" + cls_name + "/test.indoor_multi_scenario.map.box.match.txt",
        "zptest_" + cls_name: "/dataset/dataset/ssd/gesture/zptest/" + cls_name + "/zptest.map.box.match.txt",
        # "jsc220713_" + cls_name: "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/test.jsc220713.gt.txt",
        "hagrid_" + cls_name: "/dataset/dataset/ssd/gesture/hagrid/" + cls_name + "/test.hagrid.map.box.match.txt",
        "jsc220713_" + cls_name: "/dataset/dataset/ssd/gesture/jiashicang/c9/test.jsc220713.gt.txt",
        "jsc220718_day_" + cls_name: "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/test.jsc220718_day.map.box.txt",
        "jsc220715_garage_" + cls_name: "/dataset/dataset/ssd/gesture/jiashicang/" + cls_name + "/test.jsc220715_garage.map.box.txt",
    }

    det_ckp_sp = det_ckp.split("/")
    det_model_name = det_ckp_sp[-3] if det_ckp_sp[-2] == "weights" else det_ckp_sp[-2]
    det_name = det_model_name + "_" + det_ckp_sp[-1] + "_thd" + str(det_thd)
    weight_path_sp = recog_kwarg["weight_path"].split("/")
    recog_name = weight_path_sp[-3] + "_" + weight_path_sp[-1]
    test_root = "/zhoudu/test/gesture/"
    gpu_id = 2
    types = "msp"  # msp proser rodd
    eval = True
    det_use_origin_label = False  # 是否使用检测模型输出的类别替代原始的类别
    port = 50050
    is_parse_error = False
    batch_size = 32

    only_print_docker_cmd = False  # 只打印docker命令
    vis = True
    save_right = False  # 测试结果正确的是否存储然后可视化
    # cls14 测试
    # datanames = ["test_ges30_imgs_1018_1019_cls14", "test_neg_cw", "test_neg_zipai", "test_neg_imagenet"]
    # eval_datanames = {
    #
    #     "common_negcw_cls14":
    #         ["test_ges30_imgs_1018_1019_cls14", "test_neg_cw"],
    #     "common_negzipai_cls14":
    #         ["test_ges30_imgs_1018_1019_cls14", "test_neg_zipai"],
    #     "common_negimagenet_cls14":
    #         ["test_ges30_imgs_1018_1019_cls14", "test_neg_imagenet"],
    # }
    # cls7 测试
    # datanames = ["test_ges30_imgs_1018_1019_cls7", "test_indoor_multi_scenario_cls7", "test_neg_cw", "test_neg_zipai",
    #              "test_neg_imagenet"]
    # eval_datanames = {
    #     "common_negcw":
    #         ["test_ges30_imgs_1018_1019_cls7", "test_neg_cw"],
    #     "common_negzipai":
    #         ["test_ges30_imgs_1018_1019_cls7", "test_neg_zipai"],
    #     "common_negimagenet":
    #         ["test_ges30_imgs_1018_1019_cls7", "test_neg_imagenet"],
    #     "indoor_negzipai":
    #         ["test_indoor_multi_scenario_cls7", "test_neg_zipai"],
    # }
    # cls13
    # datanames = ["test_ges30_imgs_1018_1019_cls13", "test_indoor_multi_scenario_cls13", "test_neg_cw", "test_neg_zipai",
    #              "test_neg_imagenet"]
    # datanames = ["zptest_cls13"]
    # clseasy
    datanames = ["test_ges30_imgs_1018_1019_" + cls_name, "test_indoor_multi_scenario_" + cls_name,
                 "zptest_" + cls_name, "hagrid_" + cls_name, "jsc220713_" + cls_name, "jsc220718_day_" + cls_name,
                 "jsc220715_garage_" + cls_name]
    datanames = ["jsc220715_garage_" + cls_name]
    eval_datanames = {
        # "common_negcw_cls13":
        #     ["test_ges30_imgs_1018_1019_cls13", "test_neg_cw"],
        # "common_inddor_negzipai_cls13":
        #     ["test_ges30_imgs_1018_1019_cls13", "test_indoor_multi_scenario_cls13", "test_neg_zipai"],
        # "common_negimagenet_cls13":
        #     ["test_ges30_imgs_1018_1019_cls13", "test_neg_imagenet"],
        # "indoor_negzipai_cls13":
        #     ["test_indoor_multi_scenario_cls13", "test_neg_zipai"],
        # "zptest_cls13_neg":
        #     ["zptest_cls13", "test_neg_zipai"],

        "common_indoor_negzipai_" + cls_name:
            ["test_ges30_imgs_1018_1019_" + cls_name, "test_indoor_multi_scenario_" + cls_name],
        "zptest_" + cls_name:
            ["zptest_" + cls_name],
        "hagrid_" + cls_name:
            ["hagrid_" + cls_name],
        "jsc_" + cls_name:
            ["jsc220713_" + cls_name, "jsc220718_day_" + cls_name, "jsc220715_garage_" + cls_name],
    }

    for dataname in datanames:
        print("PRED: {}".format(dataname))
        save_det_path = os.path.join(test_root, "det", det_name, dataname, "det.txt")
        save_recog_path = os.path.join(test_root, "recog", det_name, recog_name, dataname, "recog.txt")
        os.makedirs(os.path.dirname(save_det_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_recog_path), exist_ok=True)

        det_src_path = datas[dataname]
        print("det_src_path: {}".format(det_src_path))
        print("save_det_path: {}".format(save_det_path))
        if not os.path.exists(save_det_path) or overwrite:
            if not use_cmd_for_det:
                extract_box(det_cfg, det_ckp, gpu_id, det_src_path, save_det_path, batch_size=batch_size, thd=det_thd,
                            is_use_origin_label=det_use_origin_label)
            else:
                cmd = "/opt/conda/bin/python /zhoudu/workspaces/yolov7/detect.py --weights " + det_ckp + " --source " + det_src_path + \
                      " --img-size 416 --conf-thres " + str(det_thd) + \
                      " --iou-thres 0.45 --contain-wh True --save-conf True" + \
                      " --save-txt-path " + save_det_path
                print("det cmd: {}".format(cmd))
                os.system(cmd)
        print("save_recog_path: {}".format(save_recog_path))
        if not os.path.exists(save_recog_path) or overwrite:
            extract_gesture(save_det_path, save_recog_path, types, gpu_id, **recog_kwarg)
    if eval:
        for eval_dataname in eval_datanames:
            print("EVAL: {}".format(eval_dataname))
            eval_names = eval_datanames[eval_dataname]
            if not only_print_docker_cmd:
                # merge gt path
                gt_path_list = []
                for eval_name in eval_names:
                    gt_path_list.append(datas[eval_name])
                if len(gt_path_list) > 1:
                    label_path_t = os.path.join("/dataset/dataset/ssd/gesture/test_list", eval_dataname, "gt.txt")
                    print("{} merge into {}".format(eval_dataname, label_path_t))
                    # if not os.path.exists(label_path_t):
                    get_merge_txt_seq(gt_path_list, label_path_t)
                    gt_path = label_path_t
                else:
                    gt_path = gt_path_list[0]

                # merge pred path
                pred_path_list = []
                for eval_name in eval_names:
                    pred_path_list.append(
                        os.path.join(test_root, "recog", det_name, recog_name, eval_name, "recog.txt"))
                if len(pred_path_list) > 1:
                    label_path_t = os.path.join(test_root, "recog", det_name, recog_name, eval_dataname, "recog.txt")
                    print("{} merge into {}".format(eval_dataname, label_path_t))
                    # if not os.path.exists(label_path_t):
                    get_merge_txt_seq(pred_path_list, label_path_t)
                    pred_path = label_path_t
                else:
                    pred_path = pred_path_list[0]

                roc_path = os.path.join(test_root, "recog", det_name, recog_name, eval_dataname, "roc.txt")
                result_info = eval_acc_and_roc(gt_path, pred_path, roc_path=roc_path, title_name=None)
                if is_parse_error:
                    parse_error(result_info)
                if vis:
                    # docker_root_dir = os.path.join(test_root, "vis", recog_name)
                    docker_root_dir = os.path.join(test_root, "vis")
                    txt_root = os.path.join(docker_root_dir, "cluster_rst")
                    txt_prefix = eval_dataname + "_(" + det_name + ")_(" + recog_name + ")_"
                    src_img_prefix = "/dataset/dataset/ssd/gesture/"
                    save_img_root = os.path.join(docker_root_dir, "imgs", det_name, recog_name)
                    if eval_dataname.startswith("jsc220713"):
                        resize = True
                        dst_size = (1080, 1080)
                    else:
                        resize = False
                        dst_size = (416, 416)
                    save_result(result_info, txt_root, txt_prefix, src_img_prefix, save_img_root, save_right=save_right,
                                resize=resize, dst_size=dst_size)
                    save_false_det(result_info, txt_root, txt_prefix, src_img_prefix, save_img_root, vis)
                    save_miss_det(result_info, txt_root, txt_prefix, src_img_prefix, save_img_root, vis)
                    server_root_dir = os.path.join("/cloudgpfs/workspace/zhoudu", docker_root_dir[1:])
                    get_docker_cmd(server_root_dir, port)
            else:
                docker_root_dir = os.path.join(test_root, "vis", recog_name)
                server_root_dir = os.path.join("/cloudgpfs/workspace/zhoudu", docker_root_dir[1:])
                get_docker_cmd(server_root_dir, port)
