# -*- coding: utf-8 -*-
# @Time    : 2021/7/19 下午3:03
# @Author  : islander
# @File    : train_global_model.py
# @Software: PyCharm

import argparse
import json
import pprint
import random
import time
from copy import deepcopy

from tensorflow.python.platform import gfile

import data
import project_path
import train_utils
import logging
import model
import config
import tensorflow as tf
import sys
import os.path as osp
import log
import numpy as np
from gutils import parse_fp

from log import hook

import log.logging_config
_logger = logging.getLogger('train_global_model')
_custom_logger = log.CustomLogger(logger=_logger)


def config_args():  # 配置命令行参数
    unparsed_args = sys.argv[1:]  # 未解析的命令行参数

    # parser 变量名按依赖顺序编号
    parser0 = argparse.ArgumentParser(add_help=False)

    run_conf_group_args = ('run_config', '运行配置，主要是传给 estimator.RunConfig 的参数')
    def regis_run_conf():
        runconf_group = parser0.add_argument_group(*run_conf_group_args)
        runconf_group.add_argument('-rands', '--tf_random_seed', default=3, type=int,
                                   help='随机数种子，除了 tensorflow，也会传给 numpy 等随机数包')
        runconf_group.add_argument('--save_summary_steps', default=1000, type=int,
                                   help='每这么多代存储 tensorflow 官方实现的一些 summary')
        runconf_group.add_argument('--save_checkpoint_secs', default=None, type=int,
                                   help='每这么长时间存储一次检查点，save_checkpoint_steps 和 save_checkpoint_secs 必须恰指定一个')
        runconf_group.add_argument('--save_checkpoint_steps', default=1000, type=int,
                                   help='每这么多迭代存储一次检查点，save_checkpoint_steps 和 save_checkpoint_secs 必须恰指定一个')
        runconf_group.add_argument('--keep_checkpoint_max', default=5, type=int,
                                   help='最多存储多少个检查点')
        runconf_group.add_argument('--keep_checkpoint_every_n_hours', default=1, type=int,
                                   help='每多少个小时保留一个检查点，保留的检查点不会因 keep_checkpoint_max 而删除')
        runconf_group.add_argument('--log_step_count_steps', default=25, type=int,
                                   help='每这么多代记录一次日志')
        runconf_group.add_argument('--device', default='gpu', type=str,
                                   help='运行设备，默认为 gpu')
        runconf_group.add_argument('-rn', '--run_name', default='train_global_model_debug', type=str,
                                   help='本次运行的任务名，打印日志有时会记录作为提示信息，'
                                        '日志将被记录在 f"{project_path.log_fd}/{run_name}"')
        runconf_group.add_argument('-nts', '--num_test_steps', default=None, type=int,
                                   help='评估时，跑多少代运算，默认评估整个测试集')
    regis_run_conf()

    def regis_pai():
        pai_group = parser0.add_argument_group('pai', 'PAI 平台自动生成的参数')
        pai_group.add_argument('--buckets', type=str, help='OSS 用户根目录')
        # 默认 job 是 worker，任务数 1，任务索引 0，符合单机训练设定，task_count/index 会影响数据集划分
        pai_group.add_argument('--job_name', default='worker', type=str,
                               choices=('worker', 'ps', 'evaluator', 'chief'),
                               help='参数服务器策略中的任务名')
        pai_group.add_argument('--task_index', default=0, type=int, help='job 内的任务 ID')
        pai_group.add_argument('--task_count', default=1, type=int, help='job 内的任务数量')
        # pai 还会传入 worker_hosts ps_hosts 等参数，脚本不需要，就不解析了
    regis_pai()

    train_group_args = ('train', '训练相关参数，如学习率')
    def regis_train():
        train_group = parser0.add_argument_group(*train_group_args)
        train_group.add_argument('-opt', '--optimizer', type=str, choices=('sgd', 'adam', 'adagrad'), default='sgd',
                                 help='使用的优化器')
        train_group.add_argument('-bs', '--batch_size', default=32, type=int,
                                 help='模型训练的 batch size')
        train_group.add_argument('--train_epoches', default=1, type=int,
                                 help='模型训练多少个 epoch')
        train_group.add_argument('--loss', default='cross_entropy', type=str, choices=('cross_entropy', 'square'),
                                 help='损失函数')
    regis_train()

    def regis_dataset():
        dataset_group = parser0.add_argument_group('dataset', '数据集相关参数')
        dataset_group.add_argument('--no_shuffle', dest='shuffle', action='store_false', default=True,
                                   help='训练时是否 shuffle 数据集，注，评估时不 shuffle')
        dataset_group.add_argument('--shuffle_cache_size', default=10000, type=int,
                                   help='shuffle 数据集时，缓存大小，缓存越大 shuffle 越均匀')
        dataset_group.add_argument('-ds', '--dataset', default='movielens', choices=('movielens', 'amazon'),
                                   help='使用的数据集')
        dataset_group.add_argument('-tdf', '--train_data_fd', type=parse_fp,
                                   default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed', 'ts=1225642324_train'),
                                   help='训练集的绝对路径，默认在 project_path.data_fd 下找')
        dataset_group.add_argument('-edf', '--eval_data_fd', type=parse_fp,
                                   default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed', 'ts=1225642324_test'),
                                   help='评估集的绝对路径，默认在 project_path.data_fd 下找')
        dataset_group.add_argument('--mapping_fp', type=parse_fp,
                                   default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed', 'movie2category.csv'),
                                   help='电影类别映射文件的路径，默认在 project_path.data_fd 下找')
        dataset_group.add_argument('--movie_genome_fp', type=parse_fp,
                                   default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'genome-scores.csv'),
                                   help='电影的硬编码 embedding 数据的路径')

    regis_dataset()

    def regis_model():
        model_group = parser0.add_argument_group('model', '模型相关参数')
        model_group.add_argument('-mo', '--model', default='din',
                                 choices=('din', 'lr', 'lr_fast', 'deepfm', 'wide_deep', 'pnn'),
                                 help='训练的机器学习模型')
        model_group.add_argument('-bn', '--batch_norm', default=None, choices=(None, 'bn'),
                                 help='是否使用 batchnorm')
    regis_model()

    args, unparsed_args = parser0.parse_known_args(args=unparsed_args, namespace=None)

    parser1 = argparse.ArgumentParser(add_help=False)

    def regis_train1():
        train_group1 = parser1.add_argument_group(*train_group_args)
        train_group1.add_argument('-lr', '--learning_rate', type=float,
                                  default={'sgd': 1.0, 'adagrad': 0.1, 'adam': 0.001}[args.optimizer],
                                  help='学习率')
        train_group1.add_argument('--batch_size_eval', default=args.batch_size, type=int,
                                  help='模型评估时的 batch size，默认与训练一致')
    regis_train1()

    run_conf_group1 = parser1.add_argument_group(*run_conf_group_args)
    run_conf_group1.add_argument('-dt', '--distribute', type=str,
                                 default='OneDeviceStrategy',
                                 choices=('OneDeviceStrategy', 'ParameterServerStrategy'),
                                 help='分布策略，默认单机训练，可选参数服务器架构训练')

    args, unparsed_args = parser1.parse_known_args(args=unparsed_args, namespace=args)

    parser_help = argparse.ArgumentParser(parents=[parser0, parser1], description='训练一个全局模型')
    parser_help.parse_known_args()

    if unparsed_args:
        _custom_logger.log_text('WARNING: Found unrecognized sys.argv: {}'.format(unparsed_args))

    return args


def main():
    entry_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # 解析命令行参数
    args = config_args()
    # 获取日志记录的目录，并创建几个相关文件
    log_fd = osp.join(project_path.log_fd, args.run_name)  # 本次运行记录日志的目录
    txt_fd = osp.join(log_fd, 'txt')  # 文本日志存放目录
    checkpoint_fd = osp.join(log_fd, 'checkpoint')  # 断点日志存放目录
    tensorboard_fd = osp.join(log_fd, 'tensorboard')  # tensorboard summary 存放目录
    # 创建几个目录
    for fd in [txt_fd, checkpoint_fd, tensorboard_fd]:
        gfile.MakeDirs(fd)
    # 这几个文件，传 None 表示不写
    is_chief = args.job_name == 'worker' and args.task_index == 0
    if is_chief:
        meta_f = gfile.GFile(osp.join(txt_fd, 'meta_{}.txt'.format(entry_time)), 'a')  # 用于记录一些运行基本信息
        train_log_f = gfile.GFile(osp.join(txt_fd, 'training_{}.csv'.format(entry_time)), 'w')  # 训练中记录动态的前向传播结果
    else:
        meta_f = None
        train_log_f = None
    if args.job_name == 'evaluator':
        eval_testset_log_f = gfile.GFile(osp.join(txt_fd, 'testset_{}.csv'.format(entry_time)), 'w')  # 记录测试集评估结果
    else:
        eval_testset_log_f = None

    def flush_all():  # 刷新所有文件
        for f in [meta_f, train_log_f, eval_testset_log_f]:
            if f is not None:
                f.flush()

    try:
        # 记录当前键入的命令
        command = log.get_command()
        _custom_logger.log_text('current command:\n{}'.format(command), file_handler=meta_f)
        # 记录处理后的命令行参数
        args_str = pprint.pformat(args.__dict__)
        _custom_logger.log_text('parsed args:\n' + args_str, file_handler=meta_f)
        if is_chief:
            # noinspection PyTypeChecker
            json.dump(args.__dict__, fp=gfile.GFile(osp.join(txt_fd, 'args_{}.json'.format(entry_time)), 'w'))  # 将参数记录下来
        _logger.info('命令行参数解析完成')

        tf.random.set_random_seed(args.tf_random_seed)
        random.seed(args.tf_random_seed)
        np.random.seed(args.tf_random_seed)
        _logger.info('随机数种子设置完成')

        estimator_config = train_utils.get_estimator_config(args=args, checkpoint_fd=checkpoint_fd)
        _logger.info('获取 estimator_config 成功')

        log_responsible_fns = gfile.GFile(osp.join(txt_fd, 'train_fns_{}.txt'.format(args.task_index)), 'w') if args.job_name == 'worker' else None
        # 确定输入配置
        if args.dataset == 'movielens':
            config_pkg = config.movielens
        elif args.dataset == 'amazon':
            config_pkg = config.amazon
        else:
            raise ValueError('Unrecognized dataset {}'.format(args.dataset))
        if args.model == 'din':
            config_pkg = config_pkg.din
        elif args.model == 'lr':
            config_pkg = config_pkg.lr
        elif args.model == 'deepfm':
            config_pkg = config_pkg.deepfm
        elif args.model == 'wide_deep':
            config_pkg = config_pkg.wide_deep
        elif args.model == 'pnn':
            config_pkg = config_pkg.pnn
        else:
            raise ValueError('unrecognized args.model = {}'.format(args.model))
        fea_config = deepcopy(config_pkg.FEA_CONFIG)
        shared_emb_config = deepcopy(config_pkg.SHARED_EMB_CONFIG)

        if args.dataset == 'movielens':
            if args.model == 'lr':
                print('loading genomes')
                movie_genomes = data.movielens.utils.load_genome(args.movie_genome_fp)
                print('genomes loaded')
            else:
                movie_genomes = None

            try:
                train_input_fn = train_utils.get_movielens_input_fn(
                    data_fd=args.train_data_fd, mapping_fp=args.mapping_fp, fea_config=fea_config, shuffle=args.shuffle,
                    shuffle_cache_size=args.shuffle_cache_size, batch_size=args.batch_size,
                    slice_count=args.task_count, slice_index=args.task_index,
                    log_responsible_fns=log_responsible_fns, movie_genome_fp=movie_genomes)
            finally:
                if log_responsible_fns is not None:
                    log_responsible_fns.close()
            eval_input_fn = train_utils.get_movielens_input_fn(
                data_fd=args.eval_data_fd, mapping_fp=args.mapping_fp, fea_config=fea_config, shuffle=False,
                batch_size=args.batch_size_eval, movie_genome_fp=movie_genomes)
        elif args.dataset == 'amazon':
            try:
                train_input_fn = train_utils.get_amazon_input_fn(
                    data_fd=args.train_data_fd, mapping_fp=args.mapping_fp, fea_config=fea_config, shuffle=args.shuffle,
                    shuffle_cache_size=args.shuffle_cache_size, batch_size=args.batch_size,
                    slice_count=args.task_count, slice_index=args.task_index, seed_plus=0,
                    log_responsible_fns=log_responsible_fns)
            finally:
                if log_responsible_fns is not None:
                    log_responsible_fns.close()
            eval_input_fn = train_utils.get_amazon_input_fn(
                data_fd=args.eval_data_fd, mapping_fp=args.mapping_fp, fea_config=fea_config, shuffle=False,
                batch_size=args.batch_size_eval, seed_plus=1000000)
        else:
            raise ValueError('Unrecognized dataset {}'.format(args.dataset))
        _logger.info('获取数据输入函数成功')

        # 构建模型
        if args.model == 'din':
            net = model.din.Din(
                input_config=fea_config,
                shared_emb_config=shared_emb_config,
                use_moving_statistics=True
            )
        elif args.model == 'lr':
            net = model.linear.LinearRegression(input_config=fea_config, fast_forward=False,
                                                batch_norm=args.batch_norm, use_moving_statistics=True)
        elif args.model == 'deepfm':
            net = model.deepfm.DeepFM(input_config=fea_config, shared_emb_config=shared_emb_config, use_moving_statistics=True)
        elif args.model == 'wide_deep':
            net = model.wide_deep.WideDeep(input_config=fea_config, shared_emb_config=shared_emb_config, use_moving_statistics=True)
        elif args.model == 'pnn':
            net = model.pnn.PNN(input_config=fea_config, shared_emb_config=shared_emb_config, use_moving_statistics=True)
        else:
            raise ValueError('unrecognized args.model = {}'.format(args.model))

        if is_chief:
            # noinspection PyTypeChecker
            json.dump({'fea_config': fea_config, 'shared_emb_config': shared_emb_config}, gfile.GFile(osp.join(txt_fd, 'config_{}.json'.format(entry_time)), 'w'))
        _logger.info('初始化模型成功')

        # 初始化 estimator，params 将会作为 model_fn 的参数，estimator_kwargs 追加为关键字参数
        # 根据给定的参数获取优化器
        optimizer = {'adam': tf.train.AdamOptimizer,
                     'sgd': tf.train.GradientDescentOptimizer,
                     'adagrad': tf.train.AdagradOptimizer}[args.optimizer](learning_rate=args.learning_rate)
        estimator = tf.estimator.Estimator(
            model_fn=net.model_fn, config=estimator_config, params={'optimizer': optimizer, 'loss': args.loss},
        )
        _logger.info('构建 estimator 成功')

        metric_names = ['auc', 'accuracy', 'false_prop', 'neg_log_loss', 'square_loss', 'num_samples', 'max_true_prob']
        if train_log_f is not None:
            content = ','.join(['global_step', *metric_names]) + '\n'
            train_log_f.write(content)
        if eval_testset_log_f is not None:
            eval_testset_log_f.write(','.join(['global_step', *metric_names]) + '\n')
        log_train_hook = hook.LogAccumulatedHook(
            tensor_name_dict=net.tensor_name_dict, metric_names=metric_names,
            file_handler=train_log_f, hint='{} training forward pass evaluation'.format(args.run_name),
            log_step_count_steps=args.log_step_count_steps)
        log_eval_hook = hook.LogAccumulatedHook(
            tensor_name_dict=net.tensor_name_dict, metric_names=metric_names,
            file_handler=eval_testset_log_f, hint='{} evaluate testset'.format(args.run_name),
            save_best_model_config={'fd': osp.join(checkpoint_fd, 'best'), 'metric_name': 'auc', 'cmp': lambda a, b: a > b})
        save_epoch_checkpoint_hook = hook.SaveEpochCheckpointHook(
            checkpoint_fd=osp.join(checkpoint_fd, 'epoch'))
        # estimator.train 会使用如下几个回调
        train_hooks = [log_train_hook, hook.LogVariableHook(file_handler=meta_f), save_epoch_checkpoint_hook]
        # estimator.evaluate 会使用如下几个回调
        eval_hooks = [log_eval_hook, hook.LogVariableHook()]
        _logger.info('创建回调完成')

        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=None, hooks=train_hooks,
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            steps=args.num_test_steps,
            start_delay_secs=0, throttle_secs=10,
            name='testset', hooks=eval_hooks,
        )
        if args.distribute == 'OneDeviceStrategy':  # 单机训练，先评估一次
            estimator.evaluate(input_fn=eval_input_fn, steps=args.num_test_steps, name='testset', hooks=eval_hooks)
        # 给定了 train_epoches，input_fn 仅循环一个 epoch，因此调用 epoch 数量次 train_and_evaluate，每次调用都是一个epoch
        flush_all()
        for epoch in range(args.train_epoches):
            _logger.info('epoch {} start'.format(epoch))
            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    finally:
        # 关闭所有文件
        def _close(f):
            if f is not None:
                f.close()
        [_close(f) for f in [train_log_f, eval_testset_log_f, meta_f]]


if __name__ == '__main__':
    main()
