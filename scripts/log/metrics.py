# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 11:06 上午
# @Author  : islander
# @File    : metrics.py
# @Software: PyCharm

"""This file implement some customized metrics operating on np.ndarray

一些统一的参数含义如下
- labels: 标签，一维，或二维但只有第一维有不等于 1，如 shape == [样本数, 1]
- probs: 预测概率，二维，shape = [样本数, 类别数]
"""

from sklearn import metrics as sklearn_metrics
import numpy as np
from collections import OrderedDict
from inspect import signature


# 数据为空
EMPTY_DATA = -1.
# 计算 AUC 时只有一个类别
AUC_ONE_CLASS = -2.
# 部分评估标准，如负样本比例，只允许二值标签
ONLY_ALLOW_BINARY = -3.
# 算 GAUC 时，没有有效（有正样本）的用户
GAUC_NO_VALID_CLIENT = -5.


def _squeeze_labels(labels):
    if len(labels.shape) != 1:
        assert len(labels.shape) == 2  # 最多只接受二维输入
        assert labels.shape[1] == 1  # 第二维应为 1（从而 reshape 后等长）
        labels = labels.squeeze(-1)
    return labels


# 计算 AUC
def auc(labels, probs):
    if len(probs) == 0:
        return EMPTY_DATA

    labels = _squeeze_labels(labels)
    # one hot
    labels_hot = np.eye(probs.shape[1], dtype=np.int32)[labels]
    try:
        auc_ = sklearn_metrics.roc_auc_score(labels_hot, probs)
    except ValueError as e:  # only one class in labels
        first_value = labels[0]
        for value in labels:
            if value != first_value:
                raise e
        # tf.logging.warning("when calculating auc, only one class in labels")
        auc_ = AUC_ONE_CLASS
    return auc_


# 计算准确率
def accuracy(labels, probs):
    if len(probs) == 0:
        return EMPTY_DATA

    labels = _squeeze_labels(labels)
    probs = np.argmax(probs, axis=1)
    return np.mean(labels == probs)


# 计算负样本比例
def false_prop(labels):
    if len(labels) == 0:
        return EMPTY_DATA

    labels = _squeeze_labels(labels)
    # ratio of false samples
    if max(labels) > 1:
        return ONLY_ALLOW_BINARY

    return 1 - labels.mean()


# 计算交叉熵损失函数
def neg_log_loss(labels, probs):
    if len(probs) == 0:
        return EMPTY_DATA

    labels = _squeeze_labels(labels)
    # negative log loss (e.g., for cross entropy)
    eps = 1e-7
    true_probs = probs[range(probs.shape[0]), labels]
    log_probs = np.log(true_probs + eps)
    return - log_probs.mean()


# 计算平方损失函数（各类别预测概率与标签差值平方之**和**，的平均值
def square_loss(labels, probs):
    if len(probs) == 0:
        return EMPTY_DATA

    labels = _squeeze_labels(labels)

    # one hot encoding labels
    labels_onehot = np.zeros(probs.shape, dtype=np.float32)
    labels_onehot[range(probs.shape[0]), labels] = 1.0

    diff = probs - labels_onehot
    diff_square = np.square(diff)
    diff_square_sum = diff_square.sum(axis=1)
    return diff_square_sum.mean()


# 计算样本数
def num_samples(labels):
    labels = _squeeze_labels(labels)
    return len(labels)


# 计算预测为正样本概率的最大值
def max_true_prob(probs):
    if len(probs) == 0:
        return EMPTY_DATA

    if probs.shape[1] != 2:
        return ONLY_ALLOW_BINARY

    return np.max(probs[:, 1])


def separate_probs(labels, probs, user_ids):
    """separate a prediction according to user_ids

    Args:
        labels: holding all labels
        probs: holding all predictions
        user_ids: holding the client id of each datum

    Returns:
        dict mapping client id to (y_true, y_pred) belonging to each client
    """
    ret = dict()

    for y_true_, y_pred_, client_ in zip(labels, probs, user_ids):
        client_ = int(client_)
        if client_ not in ret:
            ret[client_] = [[], []]
        ret[client_][0].append(y_true_)
        ret[client_][1].append(y_pred_)

    for client_ in ret:
        ret[client_] = [np.array(x) for x in ret[client_]]

    return ret


def gauc(labels, probs, user_ids):
    """compute the GAUC
    """
    if len(labels) == 0:
        return EMPTY_DATA

    labels = _squeeze_labels(labels)

    user_ids = user_ids.reshape(-1)
    if len(user_ids) != len(labels):
        raise RuntimeError(f'batch size should align, '
                           f'but when calculating GAUC, got labels={len(labels)}, user_ids={len(user_ids)}')
    separated_probs = separate_probs(labels=labels, probs=probs, user_ids=user_ids)
    auc_sum = 0.
    num_total_samples = 0
    for user_id_, (labels_, probs_) in separated_probs.items():
        auc_ = auc(labels_, probs_)
        if auc_ == EMPTY_DATA or auc_ == AUC_ONE_CLASS:
            pass
        else:
            num_client_samples = len(labels_)
            num_total_samples += num_client_samples
            auc_sum += auc_ * num_client_samples
    if num_total_samples == 0:
        return GAUC_NO_VALID_CLIENT
    else:
        return auc_sum / num_total_samples


def get_metrics(evaluation_result, metrics):
    """获取评估的各个结果

    Args:
        evaluation_result (dict): 评估结果字典，包括预测概率等值
        metrics (list): 评估指标的名称，名称应与函数名相同

    Returns:
        OrderedDict: 与 metrics 指定的顺序对应的 metric名 -> metric值 字典

    Raises:
        ValueError: evaluation_result 中缺少需要的输入，或者 metrics 对应了没有定义的函数
    """
    # 顺序记录各个 metric 和函数的对应
    metric_dict = OrderedDict()
    for metric_name in metrics:
        if metric_name not in globals():
            raise ValueError('metric {} not found'.format(metric_name))

        metric_func = globals()[metric_name]
        metric_dict[metric_name] = metric_func

    # 计算各个 metric，用字典记录到 metric_values
    metric_values = OrderedDict()
    for metric_name, metric_func in metric_dict.items():
        kwarg_names = signature(metric_func).parameters.keys()  # 获取函数的参数列表
        kwargs = dict()  # 准备填入参数键值对
        for kwarg_name in kwarg_names:
            if kwarg_name not in evaluation_result:
                raise ValueError('missing kwarg {} for metric function {}'.format(kwarg_name, metric_name))
            kwargs[kwarg_name] = evaluation_result[kwarg_name]
        metric_value = metric_func(**kwargs)  # 调用函数
        metric_values[metric_name] = metric_value

    return metric_values
