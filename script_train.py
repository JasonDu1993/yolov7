# -*- coding: utf-8 -*-
# @Time    : 2022/7/21 21:20
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : script_train.py
# @Software: PyCharm
import os
cmd = "nohup python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --workers 2" \
      " --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 256 --data data/gesture_jsc.yaml --img-size 416 416 " \
      "--cfg cfg/training/yolov7-tiny.yaml  --name yolov7_tiny_gray_md --hyp data/hyp.custom.p5.yaml " \
      "--weights /dataset/dataset/ssd/model_meta/yolov7/yolov7-tiny.pt --single-cls True --epochs 25 " \
      ">> /zhoudu/checkpoints/gesture/outs/yolov7_tiny_gray_md.txt &"
print(cmd)
# os.system(cmd)