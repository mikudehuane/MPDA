# -*- coding: utf-8 -*-
# @Time    : 2021/8/17 上午11:02
# @Author  : islander
# @File    : utils.py
# @Software: PyCharm


import tensorflow as tf


def flatten_batch(tensor):
    """将 tensor 非 batch_size 维展开

    Args:
        tensor: 待处理的 tensor

    Returns:
        处理后的 tensor
    """
    shape = tensor.shape
    feat_len = tf.reduce_prod(shape[1:])
    tensor = tf.reshape(tensor, [-1, feat_len])
    return tensor


def multi_hot(feature, depth):
    """multi-hot 编码 feature, [0, depth) 以外的值忽略，多次出现的值累加

    Examples:
        feature: [[[0, 1, -1], [0, 2, 1]],
                  [[1, 3, 4], [1, 2, 2]]]
        depth: 4
        return: [[[1, 1, 0, 0], [1, 1, 1, 0]],
                 [[0, 1, 0, 1], [0, 1, 2, 0]]]

    Args:
        depth: 编码后向量长度，即有效的实体（如电影）个数
        feature: 待编码的特征

    Returns:
        编码后的特征
    """
    # 推算输出的形状，最后一维改为 depth，用于最后 reshape
    output_shape_tailed = feature.shape.as_list()[1:]  # batch_size 以外维度的形状
    output_shape_tailed[-1] = depth  # 修改最后一维为目标深度
    output_shape = [tf.shape(feature)[0], *output_shape_tailed]

    # 变成二维 tensor（融合最后一维以外的所有维度），便于构建编码
    prefix_dim = tf.reduce_prod(tf.shape(feature)[:-1])  # 二维情况下，第一维的大小
    feature = tf.reshape(feature, [prefix_dim, feature.shape[-1]])

    def _multi_hot_one_dim(_tensor):  # multi-hot 编码一个向量
        # 去掉超出范围的内容
        _tensor = tf.boolean_mask(_tensor, tf.logical_and(_tensor >= 0, _tensor < depth))

        # 参考 https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/scatter_nd
        _update = tf.ones_like(_tensor, dtype=tf.float32)
        _indices = tf.reshape(_tensor, [-1, 1])  # 用于调用 scatter_nd 需要把最后一维变为 1
        _ret = tf.scatter_nd(_indices, _update, [depth])

        return _ret

    # 对每一个元素都做映射就是我们需要的编码效果
    feature = tf.map_fn(_multi_hot_one_dim, feature, dtype=tf.float32)

    # 改变形状到目标
    feature = tf.reshape(feature, output_shape)

    return feature
