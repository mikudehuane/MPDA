# -*- coding: utf-8 -*-
# @Time    : 2021/8/17 上午10:22
# @Author  : islander
# @File    : linear.py
# @Software: PyCharm

from ..din import ModelFrame
from ..din import constant
from ..din import utils as din_utils
import tensorflow as tf
from .. import utils


class LinearRegression(ModelFrame):
    """
    Args:
        required_feat_names: see ModelFrame
        input_config: input configuration, also see ModelFrame
        fast_forward: 是否利用稀疏更新结构快速运算

    Keyword Args:
        epsilon: arithmetic robustness
        batch_norm: batchnorm the logits
    """

    def __init__(self, input_config, required_feat_names=None, fast_forward=True, **kwargs):
        super().__init__(required_feat_names=required_feat_names, input_config=input_config, **kwargs)

        self._use_moving_statistics = kwargs.pop('use_moving_statistics', True)
        self._batch_norm = kwargs.pop('batch_norm', None)

        self._fast_forward = fast_forward

        self._seq_names = set()
        # parse input_config (input configurations)
        for feat_name, config in input_config.items():
            # for sequence inputs (val_seq or emb_seq), generate mask config
            if constant.InputConfigKeys.SEQ_NAME in config:
                seq_name = config[constant.InputConfigKeys.SEQ_NAME]
                self._seq_names.add(seq_name)
        self._seq_names = sorted(list(self._seq_names))

    def forward(self, features, mode):
        if self._fast_forward:
            return self.fast_forward(features=features, mode=mode)
        else:
            return self.basic_forward(features=features, mode=mode)

    # TODO(islander): deprecated 跑出来效果比较差，原因不明
    def fast_forward(self, features, mode):
        """通过一维 embedding 实现，数学上等价，由于稀疏性，一般更快
        """
        output = None  # 索引出的 embedding 之和，也就是 LR 的输出
        output_dim = 2  # 输出的维度，二分类，因此设为 2

        # 按顺序处理输入，并拼接
        for feat_name, feat_config in self._input_config.items():
            feature = features[feat_name]
            feat_category = feat_config[constant.InputConfigKeys.CATEGORY]

            if feat_category == constant.InputCategory.MASK:  # mask 对象，跳过，会在 seq 中处理
                pass
            elif din_utils.is_category(feat_category, constant.InputCategoryType.EMB):
                feat_emb_shape = feat_config[constant.InputConfigKeys.EMB_SHAPE]
                num_feat_entities = feat_emb_shape[0]  # 特征实体的数量，如电影的数量
                feat_default_val = feat_config.get(constant.InputConfigKeys.DEFAULT_VAL, 0)

                with tf.name_scope(feat_name):
                    feat_mask = tf.not_equal(feature, feat_default_val)  # 无论哪个位置，都要排除 padding 值
                    if din_utils.is_category(feat_category, constant.InputCategoryPlace.SEQ):
                        # 序列对象，需要结合 mask 处理，先将无效值置 -1
                        # 目前该输入维度为 [batch_size, 序列长度, 序列维度], 统一 reshape 为 [batch_size, -1] 后，multi-hot 编码
                        # 也就是会忽略 padding 值，mask 无效值后，将所有出现过的 index 填 1（多次出现 +1）

                        # 提取需要的键
                        feat_seq_name = feat_config[constant.InputConfigKeys.SEQ_NAME]
                        feat_shape = feat_config[constant.InputConfigKeys.SHAPE]

                        # 处理输入特征，无效值置换 -1
                        # 0. 从输入提取 mask
                        mask_key = constant.FEATURE_SEPARATOR.join([constant.FeaturePrefix.MASK, feat_seq_name])
                        _feat_mask = features[mask_key]
                        # 1. 将 mask 无效值转为 -1，并添加对 default_val 的 mask
                        _feat_mask = tf.equal(_feat_mask, 1, name='convert2bool')  # convert to bool tensor
                        _feat_mask = tf.expand_dims(_feat_mask, -1, name='broadcast_mask')  # [batch_size, his_len, 1]
                        _feat_mask = tf.tile(_feat_mask, [1, 1, feat_shape[-1]], name='tile_mask')  # 重复维度，使与 feature 相同
                        feat_mask = tf.logical_and(_feat_mask, feat_mask)  # 忽略 padding 值

                    # embedding 实现，每一个参数都相当于 LR 的线性层参数，把矩阵乘法转为 embedding look up
                    # embedding 维度为 2，因为有两个输出，一个对应负样本概率，一个对应正样本概率
                    emb_var = tf.get_variable('embedding/{}'.format(feat_name), shape=[num_feat_entities, output_dim], dtype=tf.float32, trainable=True)

                    # 将超出索引范围的值替换为安全值，避免 embedding_lookup 出错
                    feature = din_utils.replace_out_of_ranges(
                        value=feature, target_range=(0, num_feat_entities),
                        name='remove_invalid_emb_id', replace_value=feat_default_val)

                    feature = tf.nn.embedding_lookup(emb_var, feature, name=f'lookup_{feat_name}')

                    # 将 mask 指定参数置 0
                    # 1. 广播 mask，使与 feature 形状一致
                    _tile_multiples = [1] * len(feat_mask.shape.as_list()) + [output_dim]
                    feat_mask = tf.expand_dims(feat_mask, -1, name='broadcast_mask_to_embdim')
                    feat_mask = tf.tile(feat_mask, _tile_multiples, name='tile_mask_to_embdim')
                    # 2. 替换 0
                    feature = tf.where(feat_mask, feature, tf.zeros_like(feature), name='get_masked_feature')

                    # 求和，得到输出，保留 axis=0（batch_size）axis=-1（embedding）
                    feature_sum = tf.reduce_sum(
                        feature,
                        axis=[1, 2] if din_utils.is_category(feat_category, constant.InputCategoryPlace.SEQ) else [1]
                    )

                    # 所有特征的处理结果求和
                    if output is None:
                        output = feature_sum
                    else:
                        output = output + feature_sum
            else:
                raise NotImplementedError('LR does not currently support val type input')

        # 与 LR 的 bias 等价
        bias = tf.get_variable('bias', (output_dim,), tf.float32)
        bias = tf.reshape(bias, (1, 2))
        output = output + bias

        return output

    def basic_forward(self, features, mode):
        """按照直观上的 multihot 编码实施
        Notes:
            输入中多维输入，一律 flatten 为一维后做 multi-hot embedding
        """
        input_tensors = []  # 经过编码的输入

        with tf.name_scope('encode_input'):
            # 按顺序处理输入，并拼接
            for feat_name, feat_config in self._input_config.items():
                with tf.name_scope(feat_name):
                    feature = features[feat_name]
                    feat_category = feat_config[constant.InputConfigKeys.CATEGORY]

                    if feat_category == constant.InputCategory.MASK:  # mask 对象，跳过，会在 seq 中处理
                        pass
                    elif din_utils.is_category(feat_category, constant.InputCategoryType.EMB):
                        feat_emb_shape = feat_config[constant.InputConfigKeys.EMB_SHAPE]
                        num_feat_entities = feat_emb_shape[0]  # 特征实体的数量，如电影的数量
                        feat_default_val = feat_config.get(constant.InputConfigKeys.DEFAULT_VAL, 0)

                        feat_mask = tf.not_equal(feature, feat_default_val)  # 无论哪个位置，都要排除 padding 值
                        if din_utils.is_category(feat_category, constant.InputCategoryPlace.SEQ):
                            # 序列对象，需要结合 mask 处理，先将无效值置 -1
                            # 目前该输入维度为 [batch_size, 序列长度, 序列维度], 统一 reshape 为 [batch_size, -1] 后，multi-hot 编码
                            # 也就是会忽略 padding 值，mask 无效值后，将所有出现过的 index 填 1（多次出现 +1）

                            # 提取需要的键
                            feat_seq_name = feat_config[constant.InputConfigKeys.SEQ_NAME]
                            feat_shape = feat_config[constant.InputConfigKeys.SHAPE]

                            # 处理输入特征，无效值置换 -1
                            # 0. 从输入提取 mask
                            mask_key = constant.FEATURE_SEPARATOR.join([constant.FeaturePrefix.MASK, feat_seq_name])
                            _feat_mask = features[mask_key]
                            # 1. 将 mask 无效值转为 -1，并添加对 default_val 的 mask
                            _feat_mask = tf.equal(_feat_mask, 1, name='convert2bool')  # convert to bool tensor
                            _feat_mask = tf.expand_dims(_feat_mask, -1, name='broadcast_mask')  # [batch_size, his_len, 1]
                            _feat_mask = tf.tile(_feat_mask, [1, 1, feat_shape[-1]], name='tile_mask')  # 重复维度，使与 feature 相同
                            feat_mask = tf.logical_and(_feat_mask, feat_mask)  # 忽略 padding 值

                        feature = tf.where(feat_mask, feature, tf.fill(tf.shape(feature), -1))  # 全部无效值赋值为 -1
                        if din_utils.is_category(feat_category, constant.InputCategoryPlace.SEQ):
                            feature = utils.flatten_batch(feature)

                        feature = utils.multi_hot(feature, depth=num_feat_entities)  # [batch_size, dim, num_emb]
                        input_tensors.append(feature)
                    elif din_utils.is_category(feat_category, constant.InputCategoryType.VAL):
                        input_tensors.append(feature)
                    else:
                        raise NotImplementedError('Unsupported category {}'.format(feat_category))
            input_tensor = tf.concat(input_tensors, axis=-1, name='concat_input')

        with tf.name_scope('forward'):
            dense_layer = tf.layers.Dense(2, name='dense', trainable=True)
            input_tensor = dense_layer(input_tensor)

            if self._batch_norm is not None:
                bn = tf.layers.BatchNormalization(name='lr_bn', trainable=True)
                if self._use_moving_statistics is True:
                    bn_training = (mode == tf.estimator.ModeKeys.TRAIN)
                elif self._use_moving_statistics is False:
                    bn_training = True
                elif self._use_moving_statistics == 'always':
                    bn_training = False
                else:
                    raise ValueError('unrecognized _use_moving_statistics attribute: {}'.format(
                        self._use_moving_statistics)
                    )
                input_tensor = bn(input_tensor, training=bn_training)

        return input_tensor
