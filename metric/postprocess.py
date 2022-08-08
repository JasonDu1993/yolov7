# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 14:02
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : postprocess.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn.functional as F


def max_softmax_probability(ret):
    if isinstance(ret, torch.Tensor):
        ret = F.softmax(ret, dim=1)
        ret = ret.cpu().detach().numpy()
    conf = np.max(ret, axis=1)
    return conf


def proser_postprocessing(ret):
    if isinstance(ret, torch.Tensor):
        ret = F.softmax(ret, dim=1)
        ret = ret.cpu().detach().numpy()
    dummyconf = ret[:, -1]
    maxknownconf = np.max(ret[:, :-1], axis=1)
    conf = maxknownconf - dummyconf
    return conf


def correlation(A, B):
    corr = np.matmul(A, B)  # dims()=2, [cls_num, batch]
    if len(B.shape) == 2:
        corr /= np.linalg.norm(B, axis=0) + 1e-4
    elif len(B.shape) == 3:
        corr /= np.linalg.norm(B, axis=1)[:, None, :] + 1e-8
    corr = np.abs(corr)
    return corr


def rodd_postprocessing(ret, first_sing_vec_path):
    """

    Args:
        ret: shape [batch, fea_dim]

    Returns:

    """
    if isinstance(ret, torch.Tensor):
        ret = ret.cpu().detach().numpy()
    first_sing_vecs = np.load(first_sing_vec_path)  # shape [num_cls, fea_dim]
    corr = correlation(first_sing_vecs, ret.T)
    score = np.arccos(corr)  # score值越低越相似，因为arccos(1)=0, arccos(0)=pai/2
    if len(corr.shape) == 3:
        score = np.min(score, axis=1)
        score = np.min(score, axis=0)
    elif len(corr.shape) == 2:
        score = np.min(score, axis=0)
    score = -score  # 取相反数之后就是分数越大越相似
    return score


def postprocessing(ret, types, **kwargs):
    if types.lower() == "msp":
        conf = max_softmax_probability(ret)
    elif types.lower() == "proser":  # Learning Placeholders for Open-Set Recognition
        conf = proser_postprocessing(ret)
    elif types.lower() == "rodd":  # RODD：A Self-Supervised Approach for Robust Out-of-Distribution Detection
        first_sing_vec_path = kwargs["first_sing_vec_path"]
        conf = rodd_postprocessing(ret, first_sing_vec_path=first_sing_vec_path)
    else:
        raise Exception("not support {}".format(types))
    return conf
