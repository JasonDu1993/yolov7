import torch, cv2, random, math
import numpy as np
from PIL import Image
from torchvision import transforms
import albumentations as A


class Cutout(object):
    def __init__(self, max_holes=2, max_h_size=12, max_w_size=12, p=0.5):
        super().__init__()
        self.trans = A.CoarseDropout(max_holes, max_h_size, max_w_size, p=p)

    def __call__(self, img, bbox, kpt=None, path=None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.trans(image=img)['image']
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img, bbox, kpt, path
