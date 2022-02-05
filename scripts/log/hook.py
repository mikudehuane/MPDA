# -*- coding: utf-8 -*-
# @Time    : 2020/12/25 7:57 下午
# @Author  : islander
# @File    : hook.py
# @Software: PyCharm

"""
This file defines some customized callbacks to be inserted to the execution of estimators, typically customized logs.

The methods are:
  - begin: called before session creation
  - after_create_session: called after session creation
  - before_run: called before sess.run
    we can specified the monitored tensors in this function, and  these tensors are passed together to sess.run,
    the returned values can be got in after_run (this is useful for customized log)
  - after_run: called after sess.run
  - end: called before session destruct
"""
from collections import OrderedDict
from copy import deepcopy
from pprint import pformat

import tensorflow as tf
from tensorflow.python.platform import gfile

import os.path as osp
import numpy as np
import json
import logging
from .custom_logger import CustomLogger
from .metrics import get_metrics
from gutils import count_parameters

_logger = logging.getLogger('log.hook')
_custom_logger = CustomLogger(_logger)


def _get_session_run_args(tensor_name_dict):
    """获取 session 中指定结点的值

    Args:
        tensor_name_dict (Dict[str, str]): 自定义的键到计算图中 tensor 实际名称的映射

    Returns:
        tf.train.SessionRunArgs: tensor_name_dict.keys() 向计算图中实际 tensor 的映射
    """
    required_tensors = dict()
    for indicator, tensor_name in tensor_name_dict.items():
        op = tf.get_default_graph().get_operation_by_name(tensor_name)
        required_tensors[indicator] = op.values()[0]
    return tf.train.SessionRunArgs(required_tensors)


class LogAccumulatedHook(tf.train.SessionRunHook):
    """记录整个 session 生命周期内，预测的结果，并根据预测结果存储最佳断点

    Args:
        tensor_name_dict: 自定义的输出名 -> 计算图中实际的 tensor 名，建议传 din.tensor_name_dict
            原本变量名是在 Din.build_net 中直接获取各个 tensor 的 name 的，
            单 worker 执行没有问题，多 worker 执行发现 tensor_name_dict 是空的
            推测是因为多 worker 时，model_fn 有可能在 before_run 之后才调用，因此必须事先确定各个 tensor 的 name，
            所以现在采用的代码逻辑是事先定义各个 tensor 的 name，在 build_graph 中，
            使用 tf.identity 重命名对应的 tensor（数据输入的原始名称无法修改，并且没有明确语义）
            此外，将上一个 batch 缓存到 input_fn 里，然后读取也是不行的，因为 Dataset 异步执行
        metric_names: 需要计算的 metric 的列表
        file_handler: 算出的 metric 会写到文件，这里传入写文件的句柄
        log_step_count_steps: 每这么多代就输出一次计算的 metric 值

    Keyword Args:
        hint: 若给定，打印日志时，先输出一行提示信息
        save_best_model_config: 若指定，则会保存最优模型，包括以下 key：
            'fd': 保存最优模型的目录
            'metric_name': 保存最优模型时参考的 metric 名
            'cmp': cmp(new_result, best_result) -> True 时，更新最优模型
                best_result 会初始化为 None，因此该回调会在调用 cmp 前，判断是否为 None，如果为 None 则认为 new_result 更优

    Notes: 有两种用法
        - 用于训练中，缓存评估结果，每 n 代，计算这 n 代前向传播是算出的各种 metric，这会提示评估结果来自哪些 global_step
        - 用于评估，计算整个数据集的评估结果，这会提示当前 session 中的 global_step
    """
    def __init__(self, tensor_name_dict, metric_names, file_handler=None, *,
                 hint=None, save_best_model_config=None, log_step_count_steps=None):
        if log_step_count_steps is not None and save_best_model_config is not None:
            raise ValueError('目前 LogAccumulatedHook 只在仅计算一次 metric 是支持存储最优检查点')

        self._metric_names = metric_names
        self._tensor_name_dict = tensor_name_dict
        self._save_best_model_config = save_best_model_config
        self._hint = hint
        self._file_handler = file_handler
        self._log_step_count_steps = log_step_count_steps

        # 存储检查点相关的成员
        self._saver = None
        self._whole_outputs = dict()  # 用于缓存所有的评估结果

        self._prev_log_step = 0  # 上一次记录日志时的 global_step

    def begin(self):
        if self._save_best_model_config is not None:
            self._saver = tf.train.Saver(max_to_keep=1)

    def before_run(self, run_context):
        return _get_session_run_args(self._tensor_name_dict)

    def after_run(self, run_context: tf.train.SessionRunContext, run_values):
        outputs = run_values.results

        for key, value in outputs.items():
            try:
                value = list(value)  # this may raise error when scalar given, in this case, skip
                if key not in self._whole_outputs:
                    self._whole_outputs[key] = []
                self._whole_outputs[key].extend(value)
            except TypeError:  # not list type, e.g., loss, skip
                continue

        global_step = run_context.session.run(tf.train.get_global_step())
        if self._log_step_count_steps is not None and global_step // self._log_step_count_steps > self._prev_log_step // self._log_step_count_steps:
            self._log(global_step, hint='{}, in step [{}, {})'.format(self._hint, self._prev_log_step, global_step))
            self._whole_outputs.clear()
            self._prev_log_step = global_step

    def _log(self, global_step, hint):  # 从当前的 _whole_outputs 计算日志并清空，返回计算出的 metric 值
        logged_values = OrderedDict()  # 将被打印的值
        logged_values['global_step'] = global_step
        whole_outputs = {key: np.array(val) for key, val in self._whole_outputs.items()}  # 转为 numpy 数组
        # 根据 session 生命周期中所有的预测结果，计算 metric 的值
        metric_values = get_metrics(evaluation_result=whole_outputs, metrics=self._metric_names)
        logged_values.update(metric_values)
        _custom_logger.log_dict(msg=logged_values, hint=hint, file_handler=self._file_handler)
        return metric_values

    def end(self, session):
        global_step = int(session.run(tf.train.get_global_step()))
        metric_values = self._log(global_step, hint=self._hint)
        self._whole_outputs.clear()

        # 根据 metric 的值，判断是否保存最优模型
        if self._save_best_model_config is not None:
            # 读取配置
            checkpoint_fd = self._save_best_model_config['fd']
            metric_name = self._save_best_model_config['metric_name']
            metric_cmp = self._save_best_model_config['cmp']

            # 读取目前为止最优的 metric 值
            meta_fp = osp.join(checkpoint_fd, 'meta.json')
            best_metric = None  # 认为最差
            if gfile.Exists(meta_fp):
                meta = json.load(gfile.GFile(meta_fp, 'r'))
                best_metric = meta[metric_name]

            # 比较 metric
            current_metric = metric_values[metric_name]
            save_flag = metric_cmp(current_metric, best_metric) if best_metric is not None else True
            _logger.info('evaluate metric {}={}, previous best={}, should_save={}'.format(
                metric_name, current_metric, best_metric, save_flag))

            if save_flag:
                best_metric = current_metric
                gfile.MakeDirs(checkpoint_fd)
                save_path = self._saver.save(sess=session, save_path=osp.join(checkpoint_fd, 'model.ckpt'))
                _logger.info(f'best checkpoint saved into {save_path}')
                # noinspection PyTypeChecker
                # 存储信息
                json.dump({metric_name: best_metric, 'global_step': global_step},
                          gfile.GFile(meta_fp, 'w'))


class LogVariableHook(tf.train.SessionRunHook):
    """记录模型变量名，参数数量

    Args:
        file_handler: 记录日志时，同时会写到这个文件
    """
    def __init__(self, file_handler=None):
        self._file_handler = file_handler

    def after_create_session(self, session, coord):
        _custom_logger.log_text('global variables\n' + pformat(tf.global_variables()),
                                file_handler=self._file_handler)
        _custom_logger.log_text('trainable variables\n' + pformat(tf.trainable_variables()),
                                file_handler=self._file_handler)
        num_global_parameters = count_parameters(tf.global_variables())
        num_trainable_parameters = count_parameters(tf.trainable_variables())
        content = (f"num total parameters: {num_global_parameters}\n"
                   f"num trainable parameters: {num_trainable_parameters}\n")
        _custom_logger.log_text(content, file_handler=self._file_handler)

        if self._file_handler is not None:
            self._file_handler.flush()


class SaveEpochCheckpointHook(tf.train.SessionRunHook):
    """在 estimator 调用完毕时存储最终检查点，文件名中的 step 在 end 中调用 session.run 获得

    Args:
        checkpoint_fd: 存储检查点的目录
    """
    def __init__(self, checkpoint_fd):
        self._checkpoint_fd = checkpoint_fd
        self._saver = None

    def begin(self):
        self._saver = tf.train.Saver(max_to_keep=None)

    def end(self, session):
        global_step = session.run(tf.train.get_global_step())
        save_path = self._saver.save(sess=session, global_step=global_step,
                                     save_path=osp.join(self._checkpoint_fd, 'model.ckpt'))
        _logger.info(f'epoch checkpoint saved into {save_path}')
