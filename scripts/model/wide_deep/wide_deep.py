# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 17:15
# @Author  : islander
# @File    : wide_deep.py
# @Software: PyCharm

from ..din import ModelFrame
from ..din import constant
from ..din import utils as din_utils
import tensorflow as tf
from ..din import layers as din_layers
from ..utils import PReLU


class WideDeep(ModelFrame):
    """
    Args:
        required_feat_names: see ModelFrame
        input_config: input configuration, also see ModelFrame
        shared_emb_config: specify which features share the same embedding table

    Keyword Args:
        use_moving_statistics: whether to use moving statistics in the eval phase
    """

    def __init__(self, input_config, shared_emb_config=None, required_feat_names=None, **kwargs):
        super().__init__(required_feat_names=required_feat_names, input_config=input_config, **kwargs)

        self._use_moving_statistics = kwargs.pop('use_moving_statistics', True)

        self._emb_dict = dict()
        self._seq_names = set()

        for feat_name, config in input_config.items():
            input_category = config[constant.InputConfigKeys.CATEGORY]

            # fill _emb_dict with feat_name
            if din_utils.is_category(input_category, constant.InputCategoryType.EMB):
                self._emb_dict[feat_name] = feat_name

            # for sequence inputs (val_seq or emb_seq), generate mask config
            if constant.InputConfigKeys.SEQ_NAME in config:
                seq_name = config[constant.InputConfigKeys.SEQ_NAME]
                self._seq_names.add(seq_name)

        # replace shared embeddings in _emb_dict with emb_name
        if shared_emb_config is not None:
            for emb_name, feat_names in shared_emb_config.items():
                for feat_name in feat_names:
                    if feat_name in self._emb_dict:  # feat_names is allowed to be not contained
                        self._emb_dict[feat_name] = emb_name

        self._seq_names = sorted(list(self._seq_names))

    def __get_embeddings(self, feat_dict, emb_name2emb_variable):
        """get the corresponding embedding of the coming feat_dict, embeddings will be processed according to self._input_config

        Args:
            feat_dict: feat_name -> feat_values
            emb_name2emb_variable: the embedding dictionary

        Returns:
            the fetched embeddings
        """
        emb_feats = dict()
        for feat_name in feat_dict:
            _feat_name = feat_name.split(constant.FEATURE_SEPARATOR)[-1]
            emb_indices = feat_dict[feat_name]
            emb_name = self._emb_dict[_feat_name]
            # fetch values from feature config
            config = self._input_config[feat_name]
            emb_shape = config[constant.InputConfigKeys.EMB_SHAPE]
            default_val = config.get(constant.InputConfigKeys.DEFAULT_VAL, 0)
            emb_process = config.get(constant.InputConfigKeys.EMB_PROCESS, 'concat')

            # padded values may exceed the boundaries, force those to the configed default_val
            emb_indices = din_utils.replace_out_of_ranges(
                value=emb_indices, target_range=(0, emb_shape[0]), name='remove_invalid_emb_id', replace_value=default_val)
            embeddings = emb_name2emb_variable[emb_name]  # the embedding table for the feature
            if emb_process == 'concat':
                emb_feat = tf.nn.embedding_lookup(embeddings, emb_indices, name=f'lookup_{_feat_name}')
            else:
                assert emb_process == 'mean_skip_padding'
                # note: if no valid idx in the group given, the embedding will be reduced to zero
                emb_mask = tf.equal(emb_indices, default_val, name='get_non-padding_mask')  # convert to bool tensor
                weights = tf.where(emb_mask, tf.zeros_like(emb_mask, dtype=tf.float32),
                                   tf.ones_like(emb_mask, dtype=tf.float32), name='convert_mask2weight')  # this is to weight the embeddings shape=(*, d)
                weights_sum = tf.reduce_sum(weights, axis=-1, name='sum_weights', keepdims=True) + 1e-7  # the number of valid entries for each embedding group shape=(*, 1)
                weights = weights / weights_sum  # get the real weights (sum = 1)
                weights = tf.expand_dims(weights, -1)  # expand to the same len(shape) with emb_feat
                emb_feat = tf.nn.embedding_lookup(embeddings, emb_indices, name=f'lookup_{_feat_name}')
                emb_feat = emb_feat * weights  # apply reduce
                emb_feat = tf.reduce_sum(emb_feat, axis=-2, keepdims=True)  # reduce on the embedding group dim
            emb_feats[feat_name] = emb_feat
        return emb_feats

    def forward(self, features, mode):
        _INPUT_PARSE_FAIL_MSG = 'failure when parsing inputs for Din'

        # separate inputs
        separated_features = {
            constant.InputCategory.EMB_VEC: dict(),
            constant.InputCategory.EMB_SEQ: {seq_name: dict() for seq_name in self._seq_names},
            constant.InputCategory.EMB_TGT: dict(),
            constant.InputCategory.VAL_VEC: dict(),
            constant.InputCategory.VAL_SEQ: {seq_name: dict() for seq_name in self._seq_names},
            constant.InputCategory.VAL_TGT: dict(),
            constant.InputCategory.MASK: dict(),
        }

        with tf.variable_scope('embedding', reuse=False) as embedding_scope:  # create embedding variables, reuse=False
            # parse inputs
            for feat_name in self._input_config:
                feature = features[feat_name]
                config = self._input_config[feat_name]
                category = config[constant.InputConfigKeys.CATEGORY]
                if din_utils.is_category(category, constant.InputCategory.MASK):
                    separated_features[category][feat_name] = feature
                elif din_utils.is_category(category, constant.InputCategoryPlace.SEQ):
                    seq_name = config[constant.InputConfigKeys.SEQ_NAME]
                    separated_features[category][seq_name][feat_name] = feature
                else:
                    separated_features[category][feat_name] = feature
            # get variables
            emb_name2emb_variable = dict()
            for emb_name, _feat_name in self._get_emb_name2feat_name().items():
                feat_name = constant.FEATURE_SEPARATOR.join((constant.FeaturePrefix.FEAT, _feat_name))
                emb_shape = self._input_config[feat_name][constant.InputConfigKeys.EMB_SHAPE]
                emb_var = tf.get_variable(emb_name, shape=emb_shape, dtype=tf.float32, trainable=True)
                emb_name2emb_variable[emb_name] = emb_var

            # fetch embeddings
            emb_feat = self.__get_embeddings(separated_features[constant.InputCategory.EMB_VEC], emb_name2emb_variable)
            separated_features[constant.InputCategory.EMB_VEC] = emb_feat
            emb_feat = self.__get_embeddings(separated_features[constant.InputCategory.EMB_TGT], emb_name2emb_variable)
            separated_features[constant.InputCategory.EMB_TGT] = emb_feat
            for seq_name in self._seq_names:
                emb_feat = self.__get_embeddings(separated_features[constant.InputCategory.EMB_SEQ][seq_name],
                                                 emb_name2emb_variable)
                separated_features[constant.InputCategory.EMB_SEQ][seq_name] = emb_feat

            # concat tgt and vec
            # TODO(islander): 这里拼接时，会按照 key 排序，对于 amazon 和 movielens 来说，正好 item 和 category 是对齐的
            vec_cat = din_utils.concat_emb_val(separated_features[constant.InputCategory.EMB_VEC],
                                               separated_features[constant.InputCategory.VAL_VEC], name='concat_vec')
            seq_cat = [din_utils.concat_emb_val(separated_features[constant.InputCategory.EMB_SEQ][seq_name],
                                                separated_features[constant.InputCategory.VAL_SEQ][seq_name],
                                                name=f'concat_seq_{seq_name}')
                       for seq_name in self._seq_names]
            tgt_cat = din_utils.concat_emb_val(separated_features[constant.InputCategory.EMB_TGT],
                                               separated_features[constant.InputCategory.VAL_TGT], name='concat_tgt')

        # 拼接输入，序列输入进行 sum pool
        forward_net_inps = [vec_cat]
        ordered_masks = din_utils.get_ordered_dict_values(separated_features[constant.InputCategory.MASK])
        for seq_name, seq, mask in zip(self._seq_names, seq_cat, ordered_masks):
            sum_seq = tf.reduce_sum(seq, axis=1, name=f'sum_{seq_name}')
            forward_net_inps.append(sum_seq)
        forward_net_inps.append(tgt_cat)

        # 先做 batchnorm，统一给 FM，linear，DNN
        forward_net_inps_bned = []
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
        for forward_net_inp_idx, forward_net_inp in enumerate(forward_net_inps):
            # forward net op
            bn = tf.layers.BatchNormalization(name='forward_net_bn_{}'.format(forward_net_inp_idx), trainable=True)
            forward_net_inp = bn(forward_net_inp, training=bn_training)
            forward_net_inps_bned.append(forward_net_inp)
        forward_net_inps = forward_net_inps_bned

        # TODO(islander): this modification is for Dice, this is tricky and may introduce bug
        if self._use_moving_statistics == 'always':
            _mode = tf.estimator.ModeKeys.EVAL  # always use moving statistics, mode as train
        else:
            _mode = mode if self._use_moving_statistics else tf.estimator.ModeKeys.TRAIN

        wide_fields = forward_net_inps  # 拼接前的各个输入用来做 FM 操作
        # prepare forward net input
        forward_net_inps = tf.concat(forward_net_inps, axis=-1, name='concat_for_forward_net')

        forward_net = din_layers.MLP(layer_dims=[200, 80, 2], activations=PReLU())

        # deep part
        dnn_logits = forward_net(forward_net_inps, name='forward_net', mode=_mode)

        # wide part
        wide_inputs = [x for x in wide_fields]  # 未乘积的信息也作为输入
        with tf.name_scope('wide'):
            for field1_idx in range(len(wide_fields) - 1):
                field1 = wide_fields[field1_idx]
                for field2_idx in range(field1_idx + 1, len(wide_fields)):
                    field2 = wide_fields[field2_idx]
                    # dot product
                    wide_input = field1 * field2
                    wide_inputs.append(wide_input)
            wide_input = tf.concat(wide_inputs, axis=1, name='wide-input')
            wide_logits = tf.layers.dense(wide_input, units=2, use_bias=False)

        logits = dnn_logits + wide_logits

        return logits

    def _get_emb_name2feat_name(self):  # reverse self._emb_dict, which is feat_name -> emb_name
        emb_name2feat_name = dict()
        for feat_name, emb_name in self._emb_dict.items():
            emb_name2feat_name[emb_name] = feat_name
        return emb_name2feat_name



