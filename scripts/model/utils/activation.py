# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 17:52
# @Author  : islander
# @File    : activation.py
# @Software: PyCharm

import tensorflow as tf
from ..din import layers as din_layers


class PReLU(din_layers.AirActivation):
    """PReLU activation layer
    """
    def __init__(self, **kwargs):
        self._axis = kwargs.pop('axis', -1)

    def __call__(self, inputs, name, trainable=True, mode=tf.estimator.ModeKeys.TRAIN):
        input_shape = inputs.get_shape().as_list()

        with tf.variable_scope(name, reuse=False):
            alphas = tf.get_variable(
                'alpha', shape=[int(input_shape[self._axis])], dtype=tf.float32,
                initializer=tf.constant_initializer(0.0), trainable=trainable,
            )

            activated = tf.maximum(0.0, inputs) + alphas * tf.minimum(0.0, inputs)
            return activated
