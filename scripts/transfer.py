# -*- coding: utf-8 -*-
# @Time    : 2021/7/15 下午2:34
# @Author  : islander
# @File    : main.py
# @Software: PyCharm

"""进行迁移学习实验的入口脚本

"""
import pickle
import transfer_utils
import pandas as pd
import re
from collections import OrderedDict
from copy import deepcopy

import tensorflow as tf
import argparse
import json
import pprint
import time
import random
import numpy as np

from tensorflow.python.platform import gfile

import config
import data
import model
import project_path
import logging
import sys
import os.path as osp

import train_utils
from gutils import parse_fp, get_user_ids, get_ordered_dict, certain_hash, get_array_slice
import transfer_merge_results

import log.logging_config
from gutils import constants

_logger = logging.getLogger('transfer')
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
        runconf_group.add_argument('--device', default='gpu', type=str,
                                   help='运行设备，默认为 gpu')
        runconf_group.add_argument('-rn', '--run_name', default='transfer_debug', type=str,
                                   help='本次运行的任务名，打印日志有时会记录作为提示信息，'
                                        '日志将被记录在 f"{project_path.log_fd}/{run_name}"')
        runconf_group.add_argument('-imf', '--init_model_fp', type=parse_fp,
                                   default=osp.join(project_path.log_fd,
                                                    'train_global_model_reduce-cat_sgd_lr1-decay0.1_emb9_worker1',
                                                    'checkpoint', 'epoch', 'model.ckpt-1406271'),
                                   help='初始模型的绝对路径')

    regis_run_conf()

    def regis_pai():
        pai_group = parser0.add_argument_group('pai', 'PAI 平台自动生成的参数')
        pai_group.add_argument('--buckets', type=str, help='OSS 用户根目录')
        # task_index 会用于确认当前测试的用户
        pai_group.add_argument('-ti', '--task_index', default=0, type=int, help='job 内的任务 ID')
        pai_group.add_argument('-tc', '--task_count', default=1, type=int, help='job 内的任务数量')
        # pai 还会传入 worker_hosts ps_hosts 等参数，脚本不需要，就不解析了

    regis_pai()

    model_group_args = ('model', '模型配置')

    def regis_model():
        model_group = parser0.add_argument_group(*model_group_args)
        model_group.add_argument('-mo', '--model', default='din',
                                 choices=('din', 'lr', 'deepfm', 'wide_deep', 'pnn'),
                                 help='点击率预测模型')
        model_group.add_argument('-bn', '--batch_norm', default=None, choices=(None, 'bn'),
                                 help='是否使用 batchnorm')

    regis_model()

    train_group_args = ('train', '训练相关参数，如学习率')

    def regis_train():
        train_group = parser0.add_argument_group(*train_group_args)
        train_group.add_argument('-bs', '--batch_size', default=32, type=int,
                                 help='模型训练的 batch size')
        train_group.add_argument('--train_epoches', default=1, type=int,
                                 help='每次来外部数据，模型训练多少个 epoch')

    regis_train()

    def regis_algorithm():
        algorithm_group = parser0.add_argument_group('algorithm', '迁移学习算法相关参数')
        algorithm_group.add_argument('-mm', '--max_match', type=int, default=200,
                                     help='迁移学习算法召回用户的数量')
        algorithm_group.add_argument('-ma', '--match_algorithm', type=str, default='random',
                                     choices=('random', 'embedding', 'movie-intersection', 'movie-intersection-ratio'),
                                     help='召回算法，random 表示随机选取；embedding 表示按集中模型的 user embedding 召回；'
                                          'movie-intersection 表示与其他 user 评价过的电影取交集，按交集大小排序；'
                                          'movie-intersection-ratio 表示按交集大小/电影数量大小比例排序')
        algorithm_group.add_argument('-nax', '--not_save_as_xlsx', dest='save_as_xlsx', default=True,
                                     action='store_false', help='是否存储结果为 xlsx 文件，默认 dump 为 pkl，因为 oss 不支持')
        algorithm_group.add_argument('-se', '--skip_exp', action='store_true', default=False,
                                     help='是否跳过实验，直接拼接结果')
        algorithm_group.add_argument('-mrf', '--mid_result_fn', default='result_worker_{}.pkl',
                                     help='每个 worker 存储中间结果的文件名模板，{} 处填充 task_index')
        algorithm_group.add_argument('-uas', '--user_allocation_seed', default=0, type=int,
                                     help='分配每个 worker 的用户时，整个数组会被排序后固定随机数种子打散，'
                                          '这里指定用的随机数种子，指定不同的值可以保证数组顺序大概率不同')

    regis_algorithm()

    def regis_dataset():
        dataset_group = parser0.add_argument_group('dataset', '数据集相关参数')
        dataset_group.add_argument('-ds', '--dataset', default='movielens', choices=('movielens', 'amazon'),
                                   help='使用的数据集')
        dataset_group.add_argument('--movie_genome_fp', type=parse_fp,
                                   default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'genome-scores.csv'),
                                   help='电影的硬编码 embedding 数据的路径')
        dataset_group.add_argument('-tdf', '--train_data_fd', type=parse_fp,
                                   default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed',
                                                    'ts=1225642324_train'),
                                   help='训练集的绝对路径，默认在 project_path.data_fd 下找，召回用户的范围是该文件夹下的用户')
        dataset_group.add_argument('-edf', '--eval_data_fd', type=parse_fp,
                                   default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed',
                                                    'ts=1225642324_test'),
                                   help='评估集的绝对路径，默认在 project_path.data_fd 下找')
        dataset_group.add_argument('-eulp', '--examined_user_list_fp', type=parse_fp,
                                   default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed',
                                                    'ts=1225642324_user-intersect.json'),
                                   help='要给哪些用户跑算法，json 格式的文件，记录 user_id 的列表')
        dataset_group.add_argument('--mapping_fp', type=parse_fp,
                                   default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed',
                                                    'movie2category.csv'),
                                   help='电影类别映射文件的路径，默认在 project_path.data_fd 下找')
        dataset_group.add_argument('-aup', '--all_users_fp', type=parse_fp, default=None,
                                   help='json 格式文件，指定被共享数据的用户全集，默认为 train_data_fd 下全部用户')

    regis_dataset()

    args, unparsed_args = parser0.parse_known_args(args=unparsed_args, namespace=None)

    parser1 = argparse.ArgumentParser(add_help=False)

    def regis_train1():
        train_group1 = parser1.add_argument_group(*train_group_args)
        train_group1.add_argument('-lr', '--learning_rate', type=float,
                                  default=0.1, help='学习率，因为是微调，比集中训练设的小')
        train_group1.add_argument('--batch_size_eval', default=args.batch_size, type=int,
                                  help='模型评估时的 batch size，默认与训练一致')

    regis_train1()

    def regis_model1():
        model_group = parser1.add_argument_group(*model_group_args)
        if args.model == 'din':
            model_group.add_argument('-fe', '--freeze_embeddings', default=False, action='store_true',
                                     help='是否冻结 embedding 层')
            model_group.add_argument('-fbn', '--freeze_bn', default=False, action='store_true',
                                     help='是否冻结 batchnorm 层')
            model_group.add_argument('-ffn', '--freeze_forward_net', default=False, action='store_true',
                                     help='是否冻结 forward_net 层')
            model_group.add_argument('-fa', '--freeze_attention', default=False, action='store_true',
                                     help='是否冻结 attention 层')
        elif args.model == 'lr':
            pass
        elif args.model == 'deepfm':
            pass
        elif args.model == 'wide_deep':
            pass
        elif args.model == 'pnn':
            pass
        else:
            raise ValueError('Unrecognized model {}'.format(args.model))

    regis_model1()

    args, unparsed_args = parser1.parse_known_args(args=unparsed_args, namespace=args)

    parser_help = argparse.ArgumentParser(parents=[parser0, parser1], description='迁移学习算法')
    parser_help.parse_known_args()

    if unparsed_args:
        _custom_logger.log_text('WARNING: Found unrecognized sys.argv: {}'.format(unparsed_args))

    return args


def run_exp_for_a_user(args, user_id, msg, *, movie2categories=None, match_func=None, movie_genomes=None):
    """运行 user_id 对应用户的实验

    Args:
        movie_genomes: 电影基因组数据
        match_func: 召回函数，match_func.get_match(user_id) 返回被召回的用户列表
        args: 解析后的命令行参数
        msg: 传入的信息，在每个用户检查完后的日志开头打印
        user_id (str): 当前实验用户的 id

        以下参数都是外循环预处理出来的 python 对象，用于避免重复运算，理论上都可以直接从 args 和 user_id 计算
        movie2categories (Dict[str, List[str]]): 全部参数都有；电影到类别的映射

    Returns:
        记录日志的目录
    """

    def flush(_f):  # 刷新文件
        if _f is not None:
            _f.flush()

    def flush_all():  # 刷新所有文件
        for f in [meta_f, eval_trainset_log_f, eval_testset_log_f]:
            flush(f)

    def _pure_eval_local_and_save_log(_step):
        """评估本地训练集和测试集并保存日志，不更新任何全局变量（如 best_metric）

        Args:
            _step: 记录日志时 step 列填写的信息
        """
        # 评估训练集
        _metric_values = transfer_utils.get_metric_values(net=net, input_fn=local_train_input_fn,
                                                          metric_names=metric_names, graph_to_eval='train')
        _metric_values.update({'step': _step, 'is_current_user_selected': None,
                               'num_selected_users': None, 'best_metric': None})
        _custom_logger.log_dict(msg=get_ordered_dict(_metric_values, columns_order), file_handler=eval_trainset_log_f,
                                hint='{} model evaluated on local trainset'.format(_step))
        # 评估测试集
        _metric_values = transfer_utils.get_metric_values(net=net, input_fn=local_eval_input_fn,
                                                          metric_names=metric_names, graph_to_eval='train')
        _metric_values.update({'step': _step, 'is_current_user_selected': None,
                               'num_selected_users': None, 'best_metric': None})
        _custom_logger.log_dict(msg=get_ordered_dict(_metric_values, columns_order), file_handler=eval_testset_log_f,
                                hint='{} model evaluated on local testset'.format(_step))

    def _get_input_fn(_fp, *, _seed_plus=None, _batch_size):  # 根据指定的数据集调用对应的函数，获得数据输入对象
        if args.dataset == 'movielens':
            # movie_genomes_fp 当模型为 Din 时为 None，该函数可以处理
            _input_fn = train_utils.get_movielens_input_fn(
                _fp, movie2categories,
                require='generator', fea_config=fea_config, batch_size=_batch_size,
                movie_genome_fp=movie_genomes)
        elif args.dataset == 'amazon':
            assert _seed_plus is not None
            # movie2categories 在 amazon 数据集中表示 item 到类别的映射，这里就不修改变量名了
            _input_fn = train_utils.get_amazon_input_fn(
                _fp, movie2categories,
                require='generator', fea_config=fea_config, batch_size=_batch_size,
                seed_plus=_seed_plus)
        else:
            raise ValueError('Unrecognized dataset {}'.format(args.dataset))
        return _input_fn

    entry_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    _logger.info('examine user {}, entry_time={}'.format(user_id, entry_time))
    # 获取日志记录的目录，并创建几个相关文件
    log_fd = osp.join(project_path.log_fd, args.run_name, user_id)  # 本次运行记录日志的目录，记录在以用户 id 命名的子目录
    if gfile.Exists(log_fd):  # 日志目录已经存在，说明跑过这个用户了
        raise RuntimeError(
            'attempt to run experiment for user {}, but found a previous run in {}'.format(user_id, log_fd))
    txt_fd = log_fd  # 文本日志存放目录，避免过多目录删除耗时，存在日志子目录根目录，如果需要存断点，之后再创目录
    # 创建几个目录
    gfile.MakeDirs(txt_fd)
    meta_f = gfile.GFile(osp.join(txt_fd, 'meta.txt'), 'w')  # 用于记录一些运行基本信息
    eval_trainset_log_f = gfile.GFile(osp.join(txt_fd, 'trainset.csv'), 'w')  # 算法每一步会在训练集上评估
    eval_testset_log_f = gfile.GFile(osp.join(txt_fd, 'testset.csv'), 'w')  # 如果模型更新了，会在测试集上评估结果

    meta_names = ['step', 'is_current_user_selected', 'num_selected_users', 'best_metric']  # 表头中其他列的名称
    # 评估时记录的标准

    metric_names = ['auc', 'accuracy', 'false_prop', 'neg_log_loss', 'square_loss', 'num_samples', 'max_true_prob']
    columns_order = meta_names + metric_names  # 列名的顺序
    try:
        # 记录时间戳，和表头
        _custom_logger.log_text('entry_time: {}'.format(entry_time), file_handler=meta_f)
        heading = ','.join(meta_names + metric_names) + '\n'
        eval_trainset_log_f.write(heading)
        eval_testset_log_f.write(heading)
        flush_all()

        tf.random.set_random_seed(args.tf_random_seed)
        random.seed(args.tf_random_seed)
        np.random.seed(args.tf_random_seed)
        _logger.info('随机数种子设置完成')

        # 根据命令行参数提取配置
        # 首先判断数据集类型
        if args.dataset == 'movielens':
            config_pkg = config.movielens
        elif args.dataset == 'amazon':
            config_pkg = config.amazon
        else:
            raise ValueError('Unrecognized dataset {}'.format(args.dataset))
        # 然后判断模型类型
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

        # 构建模型
        if args.model == 'din':
            net = model.din.Din(
                input_config=fea_config,
                shared_emb_config=shared_emb_config,
                use_moving_statistics='always'
            )
            if args.freeze_embeddings:
                net.freeze_embeddings()
            if args.freeze_bn:
                net.freeze_bn()
            if args.freeze_forward_net:
                net.freeze_forward_net()
            if args.freeze_attention:
                net.freeze_attention()
        elif args.model == 'lr':
            net = model.linear.LinearRegression(
                input_config=fea_config, fast_forward=False, batch_norm=args.batch_norm, use_moving_statistics='always'
            )
        elif args.model == 'deepfm':
            net = model.deepfm.DeepFM(input_config=fea_config, shared_emb_config=shared_emb_config,
                                      use_moving_statistics='always')
        elif args.model == 'wide_deep':
            net = model.wide_deep.WideDeep(input_config=fea_config, shared_emb_config=shared_emb_config,
                                           use_moving_statistics='always')
        elif args.model == 'pnn':
            net = model.pnn.PNN(input_config=fea_config, shared_emb_config=shared_emb_config,
                                use_moving_statistics='always')
        else:
            raise ValueError('Unrecognized model {}'.format(args.model))
        # 构建训练图并加载初始模型
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
        net.build_graph_(key='train', mode=tf.estimator.ModeKeys.TRAIN, device=args.device,
                         optimizer=optimizer, seed=args.tf_random_seed)
        net.switch_graph('train')
        net.saver.restore(net.session, args.init_model_fp)
        net.build_graph_(key='eval', mode=tf.estimator.ModeKeys.EVAL, device=args.device)
        net.load_from_to(from_key='train', to_key='eval')
        net.build_graph_(key='selected', mode=tf.estimator.ModeKeys.EVAL, device='cpu')  # 被选择的模型，放 cpu 上，减少显存占用
        net.load_from_to(from_key='train', to_key='selected')
        _logger.info('模型构建并加载初始参数完成')

        matched_user_ids = match_func.get_match(user_id=user_id)
        _custom_logger.log_text('matched users:\n{}'.format(matched_user_ids), file_handler=meta_f)
        flush(meta_f)
        _logger.info('召回用户列表确认')

        local_train_input_fn = _get_input_fn(osp.join(args.train_data_fd, '{}.csv'.format(user_id)),
                                             _seed_plus=constants.AMAZON_TRAINSET_SEED_PLUS,
                                             _batch_size=args.batch_size)
        local_eval_input_fn = _get_input_fn(osp.join(args.eval_data_fd, '{}.csv'.format(user_id)),
                                            _seed_plus=constants.AMAZON_TESTSET_SEED_PLUS,
                                            _batch_size=args.batch_size_eval)
        _logger.info('本地数据读取的生成器创建完成')

        # 评估本地训练的模型
        net.switch_graph('train')
        net.load_from('selected')
        # 本地数据训练
        for epoch in range(args.train_epoches):
            local_train_input_fn.reader.seek(0)
            model.train_by_net(net=net, input_fn=local_train_input_fn, train_steps=None)
        # 评估训练集
        metric_values = transfer_utils.get_metric_values(net=net, input_fn=local_train_input_fn,
                                                         metric_names=metric_names,
                                                         graph_to_eval='train')
        metric_values.update({'step': 'local_data_trained', 'is_current_user_selected': None,
                              'num_selected_users': None, 'best_metric': None})
        _custom_logger.log_dict(msg=get_ordered_dict(metric_values, columns_order), file_handler=eval_trainset_log_f,
                                hint='examining user {}, local-data-trained model evaluated on local trainset'.format(
                                    user_id))
        # 评估测试集
        metric_values = transfer_utils.get_metric_values(net=net, input_fn=local_eval_input_fn,
                                                         metric_names=metric_names,
                                                         graph_to_eval='train')
        metric_values.update({'step': 'local_data_trained', 'is_current_user_selected': None,
                              'num_selected_users': None, 'best_metric': None})
        _custom_logger.log_dict(msg=get_ordered_dict(metric_values, columns_order), file_handler=eval_testset_log_f,
                                hint='examining user {}, local-data-trained model evaluated on local testset'.format(
                                    user_id))
        # 还原模型
        net.load_from_to(from_key='selected', to_key='train')

        # 评估初始模型
        # 评估训练集
        metric_values = transfer_utils.get_metric_values(net=net, input_fn=local_train_input_fn,
                                                         metric_names=metric_names,
                                                         graph_to_eval='train')
        best_metric = metric_values['auc']
        metric_values.update({'step': 'init_model', 'is_current_user_selected': None,
                              'num_selected_users': 0, 'best_metric': best_metric})
        _custom_logger.log_dict(msg=get_ordered_dict(metric_values, columns_order), file_handler=eval_trainset_log_f,
                                hint='initial model evaluated on local trainset')
        # 评估测试集
        metric_values = transfer_utils.get_metric_values(net=net, input_fn=local_eval_input_fn,
                                                         metric_names=metric_names,
                                                         graph_to_eval='train')
        metric_values.update({'step': 'init_model', 'is_current_user_selected': None,
                              'num_selected_users': 0, 'best_metric': best_metric})
        _custom_logger.log_dict(msg=get_ordered_dict(metric_values, columns_order), file_handler=eval_testset_log_f,
                                hint='initial model evaluated on local testset')

        selected_user_ids = []  # 被选择的用户
        for external_user_step, external_user_id in enumerate(matched_user_ids):
            external_user_input_fn = _get_input_fn(osp.join(args.train_data_fd, '{}.csv'.format(external_user_id)),
                                                   _seed_plus=constants.AMAZON_TRAINSET_SEED_PLUS,
                                                   _batch_size=args.batch_size)

            # 还原模型
            net.switch_graph('train')
            net.load_from('selected')  # 加载当前的最优模型

            # 用外部数据训练 n 个 epoch
            for epoch in range(args.train_epoches):
                external_user_input_fn.reader.seek(0)
                model.train_by_net(net=net, input_fn=external_user_input_fn, train_steps=None)

            # 用本地训练数据验证结果
            metric_values = transfer_utils.get_metric_values(net=net, input_fn=local_train_input_fn,
                                                             metric_names=metric_names, graph_to_eval='train')
            is_selected = metric_values['auc'] > best_metric + 1e-7  # +1e-7 避免相同的值因为浮点数表示被认为大了
            # 获取应该被记录的日志
            if is_selected:
                best_metric = metric_values['auc']
                selected_user_ids.append(external_user_id)
            # 记录训练结果
            meta_columns = {'step': external_user_step, 'is_current_user_selected': is_selected,
                            'num_selected_users': len(selected_user_ids), 'best_metric': best_metric}
            metric_values.update(meta_columns)
            if msg is not None:
                _logger.info(msg)
            _custom_logger.log_dict(msg=get_ordered_dict(metric_values, columns_order),
                                    file_handler=eval_trainset_log_f,
                                    hint='rn={}, model trained by {}, evaluated on user {}\' local trainset'.format(
                                        args.run_name, external_user_id, user_id))
            # 更新模型等操作
            if is_selected:
                net.load_from_to(from_key='train', to_key='selected')  # 记录最优模型
                # 用本地测试数据验证结果
                metric_values = transfer_utils.get_metric_values(net=net, input_fn=local_eval_input_fn,
                                                                 metric_names=metric_names, graph_to_eval='selected')
                metric_values.update(meta_columns)
                _custom_logger.log_dict(msg=get_ordered_dict(metric_values, columns_order),
                                        file_handler=eval_testset_log_f,
                                        hint='rn={}, model trained by {}, evaluated on user {}\' local testset'.format(
                                            args.run_name, external_user_id, user_id))
        # noinspection PyTypeChecker
        json.dump(selected_user_ids, gfile.GFile(osp.join(txt_fd, 'selected_user_ids.json'), 'w'))

        # 用本地数据训练模型
        net.switch_graph('train')
        net.load_from('selected')
        # 本地数据训练
        for epoch in range(args.train_epoches):
            local_train_input_fn.reader.seek(0)
            model.train_by_net(net=net, input_fn=local_train_input_fn, train_steps=None)
        _pure_eval_local_and_save_log(_step='local&external_data_trained')

        # 消融实验，直接用召回的用户序列训练模型
        net.switch_graph('train')
        net.saver.restore(net.session, args.init_model_fp)
        # 直接用召回用户的前 N 个训练模型
        for external_user_step, external_user_id in enumerate(matched_user_ids[:len(selected_user_ids)]):
            external_user_input_fn = _get_input_fn(osp.join(args.train_data_fd, '{}.csv'.format(external_user_id)),
                                                   _seed_plus=constants.AMAZON_TRAINSET_SEED_PLUS,
                                                   _batch_size=args.batch_size)

            # 用外部数据训练 n 个 epoch
            for epoch in range(args.train_epoches):
                external_user_input_fn.reader.seek(0)
                model.train_by_net(net=net, input_fn=external_user_input_fn, train_steps=None)

            _logger.info('train directly on matched users, {}/{} users passed'.format(
                external_user_step + 1, len(selected_user_ids)))
        # 同样在最后进行一次本地训练
        for epoch in range(args.train_epoches):
            local_train_input_fn.reader.seek(0)
            model.train_by_net(net=net, input_fn=local_train_input_fn, train_steps=None)
        _pure_eval_local_and_save_log(_step='vanilla_match')

    finally:
        # 关闭所有文件
        def _close(f):
            if f is not None:
                f.close()

        [_close(f) for f in [eval_trainset_log_f, eval_testset_log_f, meta_f]]

    return log_fd


def is_log_complete(log_fd):
    if not log_fd.endswith(osp.sep):
        log_fd = log_fd + osp.sep
    if gfile.Exists(log_fd):
        trainset_fp = osp.join(log_fd, 'trainset.csv')
        testset_fp = osp.join(log_fd, 'testset.csv')

        def _check_for_file(_fp):
            _is_file_complete = False  # 文件是否完整初始化为 False

            if gfile.Exists(_fp):
                # noinspection PyTypeChecker
                _fp_data = pd.read_csv(gfile.GFile(_fp), dtype=str)
                for idx, row in _fp_data.iterrows():
                    # TODO(islander): 当增删实验项目时需要修改
                    if row.step == 'vanilla_match':  # 文件最后一行的 step 为这个
                        _is_file_complete = True
            return _is_file_complete

        return _check_for_file(trainset_fp) and _check_for_file(testset_fp)
    else:
        return False


def main():
    entry_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    args = config_args()

    # 获取日志记录的目录
    log_fd = osp.join(project_path.log_fd, args.run_name)  # 本次运行记录日志的根目录
    # 创建几个目录
    gfile.MakeDirs(log_fd)

    # 记录命令到日志根目录
    if args.all_users_fp is None:
        # 所有可能被选为数据源的用户，仅考虑有训练集的用户
        all_user_ids = get_user_ids([args.train_data_fd])
    else:
        all_user_ids = json.load(gfile.GFile(args.all_users_fp))
    if args.task_index == 0:
        with gfile.GFile(osp.join(log_fd, 'meta_{}.txt'.format(entry_time)), 'w') as meta_f:  # 用于记录一些运行基本信息
            # 记录当前键入的命令
            command = log.get_command()
            _custom_logger.log_text('current command:\n{}'.format(command), file_handler=meta_f)
            # 记录处理后的命令行参数
            args_str = pprint.pformat(args.__dict__)
            _custom_logger.log_text('parsed args:\n' + args_str, file_handler=meta_f)
            # noinspection PyTypeChecker
            json.dump(args.__dict__,
                      fp=gfile.GFile(osp.join(log_fd, 'args_{}.json'.format(entry_time)), 'w'))  # 将参数记录下来
            _custom_logger.log_text('all_user_ids:\n{}'.format(all_user_ids), file_handler=meta_f)
    _logger.info('命令行参数解析完成')

    # worker 存储结果的目录，如果已经存在，说明 worker 已经结束了
    dump_fp = osp.join(log_fd, args.mid_result_fn.format(args.task_index))
    is_worker_result_got = gfile.Exists(dump_fp)

    # 所有被实验的用户
    all_examined_user_ids = json.load(gfile.GFile(args.examined_user_list_fp, 'r'))
    # 当前 worker 负责哪些用户
    user_id_slices = get_array_slice(all_examined_user_ids, slice_count=args.task_count,
                                     order_seed=args.user_allocation_seed)
    user_ids = user_id_slices[args.task_index]
    _logger.info('responsible users for current worker:\n{}'.format(user_ids))

    if not args.skip_exp and not is_worker_result_got:
        movies2categories = data.movielens.utils.load_category_mapping(args.mapping_fp)
        if args.dataset == 'movielens' and args.model == 'lr':
            print('loading genomes')
            movie_genomes = data.movielens.utils.load_genome(args.movie_genome_fp)
            print('genomes loaded')
        else:
            movie_genomes = None

        result_fds = []

        if args.match_algorithm == 'random':
            # 这里设置种子，让召回器使用内部的随机数生成器，从而所有用户共用该随机数生成器，避免召回相同的用户
            match_func = transfer_utils.match.RandomMatch(all_user_ids, seed=args.tf_random_seed,
                                                          max_match=args.max_match)
        elif args.match_algorithm == 'movie-intersection':
            match_func = transfer_utils.match.MovieIntersectionMultiWorkerMatch(
                all_user_ids, data_fd=args.train_data_fd, max_match=args.max_match,
                task_index=args.task_index, task_count=args.task_count, log_fd=log_fd)
        elif args.match_algorithm == 'movie-intersection-ratio':
            match_func = transfer_utils.match.MovieIntersectionMultiWorkerMatch(
                all_user_ids, data_fd=args.train_data_fd, max_match=args.max_match,
                task_index=args.task_index, task_count=args.task_count, log_fd=log_fd,
                order_by_ratio=True
            )
        else:
            raise ValueError('args.match_algorithm={}, unrecognized'.format(args.match_algorithm))
        _logger.info('召回函数索引构建完成')

        for user_id_idx, user_id in enumerate(user_ids):
            result_fd = osp.join(log_fd, user_id)
            if is_log_complete(result_fd):
                _logger.info('user {} has been examined, skip'.format(user_id))
            else:
                if gfile.Exists(result_fd):
                    _logger.info('user {} is not complete'.format(user_id))
                    gfile.DeleteRecursively(result_fd)
                else:
                    _logger.info('user {} has not been examined'.format(user_id))

                run_exp_for_a_user(args, user_id=user_id, movie_genomes=movie_genomes,
                                   msg='global progress: {}/{}'.format(user_id_idx, len(user_ids)),
                                   movie2categories=movies2categories, match_func=match_func)

            result_fds.append(result_fd)
            _logger.info('{}/{} users examined'.format(user_id_idx + 1, len(user_ids)))
    else:
        result_fds = [osp.join(log_fd, user_id) for user_id in user_ids]

    if not is_worker_result_got:
        # 存储实验结果，融合到一个文件
        # 将本 worker 产生的所有结果融合到一个 ndarray
        result_arrs = transfer_merge_results.merge_into_array(result_fds)
        # noinspection PyTypeChecker
        pickle.dump(result_arrs, gfile.GFile(dump_fp, 'wb'))

    # 判断其他 worker 退出了没有
    fns = gfile.ListDirectory(log_fd)
    pat = re.compile(args.mid_result_fn.replace('{}', r'(\d+)').replace('.', r'\.'))
    fns = [fn for fn in fns if re.match(pat, fn)]
    exited_task_indices = set([int(re.match(pat, fn).group(1)) for fn in fns])  # 已经完成的任务索引
    remaining_task_indices = [task_index for task_index in range(args.task_count) if
                              task_index not in exited_task_indices]  # 除去之后得到没有推出的
    is_all_mapper_finished = not bool(remaining_task_indices)  # 剩余为空，说明都退出了，task_count 可以 < 结果数量，发生在使用 --skip_mapping 时

    # 如果全部 worker 都结束了，融合文件为一个 numpy 对象
    if is_all_mapper_finished:
        # 存放结果的文件列表
        result_fps = [osp.join(log_fd, args.mid_result_fn.format(task_index))
                      for task_index in range(args.task_count)]
        # noinspection PyTypeChecker
        arrays = [pickle.load(gfile.GFile(result_fp, 'rb')) for result_fp in result_fps]
        # 所有 worker 整合出的结果
        full_arr = transfer_merge_results.merge_arrs_from_workers(arrays, sorted_by_uid=True)

        if args.save_as_xlsx:
            result_fp = osp.join(log_fd, 'result_{}.xlsx'.format(entry_time))
            workbook = transfer_merge_results.convert_arrays_to_workbook(full_arr)
            workbook.save(result_fp)
        else:
            result_fp = osp.join(log_fd, 'result_{}.pkl'.format(entry_time))
            # noinspection PyTypeChecker
            pickle.dump(full_arr, gfile.GFile(result_fp, 'wb'))


if __name__ == '__main__':
    main()
