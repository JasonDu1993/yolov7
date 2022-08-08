# -*- coding: utf-8 -*-
# @Time    : 2022/6/24 15:26
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : roc.py
# @Software: PyCharm
import numpy as np
import warnings


def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    axis : int, optional
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                             atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out


def binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification

    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function

    pos_label : int or str, default=None
        The label of the positive class

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).

    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).

    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """

    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def get_roc_with_err_rates(y_true, y_score, pos_label=1, err_rates=None):
    if err_rates is None:
        err_rates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    roc = []
    recalls = []
    pos_index = (y_true == pos_label)
    pos_num = np.sum(pos_index)
    pos_sco = y_score[pos_index]
    pos_sco_ind = np.argsort(pos_sco, kind="mergesort")[::-1]  # 从大到小排序
    pos_sco_sort = pos_sco[pos_sco_ind]

    neg_index = (y_true != pos_label)
    neg_num = np.sum(neg_index)
    neg_sco = y_score[neg_index]
    neg_sco_ind = np.argsort(neg_sco, kind="mergesort")[::-1]
    neg_sco_sort = neg_sco[neg_sco_ind]

    for err_rate in err_rates:
        err_num = int(neg_num * err_rate)
        threshold = neg_sco_sort[err_num]
        r = np.sum(pos_sco_sort > threshold)
        recall = r / pos_num
        recalls.append(recall)
        precision = r / (r + err_num)
        roc.append([err_rate, recall, precision, threshold])
        # print("err_rate:{} recall:{:.4f} threshold:{:.4f}".format(err_rate, float(recall), threshold))
    return roc


def get_fprs_with_tprs(y_true, y_score, pos_label=1, tprs=None):
    if tprs is None:
        tprs = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6]
    roc = []
    fprs = []
    pos_index = (y_true == pos_label)
    pos_num = np.sum(pos_index)
    pos_sco = y_score[pos_index]
    pos_sco_ind = np.argsort(pos_sco, kind="mergesort")[::-1]  # 从大到小排序
    pos_sco_sort = pos_sco[pos_sco_ind]

    neg_index = (y_true != pos_label)
    neg_num = np.sum(neg_index)
    neg_sco = y_score[neg_index]
    neg_sco_ind = np.argsort(neg_sco, kind="mergesort")[::-1]
    neg_sco_sort = neg_sco[neg_sco_ind]

    for tpr in tprs:
        tp_num = int(pos_num * tpr)
        threshold = pos_sco_sort[tp_num]
        f = np.sum(neg_sco_sort > threshold)
        fpr = f / neg_num
        fprs.append(fpr)
        precision = tp_num / (tp_num + f)
        roc.append([fpr, tpr, precision, threshold])
        # print("err_rate:{} recall:{:.4f} threshold:{:.4f}".format(err_rate, float(recall), threshold))
    return roc
