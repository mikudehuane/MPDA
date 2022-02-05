# -*- coding: utf-8 -*-
# @Time    : 2021/7/22 下午2:13
# @Author  : islander
# @File    : eval_global_model.py
# @Software: PyCharm

import argparse
import csv
import json
import pprint
import re
import struct
from copy import deepcopy

from tensorflow.python.platform import gfile

import data
import project_path
import os.path as osp
import config
import model
import logging
import tensorflow as tf

import log.logging_config
import train_utils
from gutils import parse_fp
import numpy as np

_logger = logging.getLogger('eval_global_model')
_custom_logger = log.CustomLogger(logger=_logger)

PROTOCOL = '>qddq'  # 写二进制文件的协议 long int (user id), double (prob[0]), double (prob[1]), long int (label)
NUM_BYTES = 32


def config_args():
    parser = argparse.ArgumentParser('评估一个模型在一个数据集上的效果')

    parser.add_argument('-mf', '--model_fp', type=parse_fp,
                        help='模型检查点的绝对路径')
    parser.add_argument('-df', '--data_fd', type=parse_fp,
                        default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed',
                                         'ts=1225642324_test'),
                        help='待评估数据存放目录的绝对路径')
    parser.add_argument('--device', default='gpu', type=str, help='运行设备，默认为 gpu')
    parser.add_argument('--mapping_fp', type=parse_fp,
                        default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed',
                                         'movie2category.csv'),
                        help='电影类别映射文件的绝对路径，默认在 project_path.data_fd 下找')
    parser.add_argument('-ti', '--task_index', default=0, type=int, help='job 内的任务 ID')
    parser.add_argument('-tc', '--task_count', default=1, type=int, help='job 内的任务数量')
    parser.add_argument('-nts', '--num_test_steps', default=None, type=int,
                        help='评估时，跑多少代运算，默认评估整个测试集')
    parser.add_argument('-rn', '--run_name', default='evaluate_global_model_debug', type=str,
                        help='本次运行的任务名，打印日志有时会记录作为提示信息，'
                             '日志将被记录在 f"{project_path.log_fd}/{run_name}"')
    parser.add_argument('--skip_mapping', action='store_true', default=False,
                        help='是否跳过预测的过程，直接 reduce 计算')
    parser.add_argument('-lpf', '--log_parent_fd', default=project_path.log_fd, type=parse_fp,
                        help='日志将被存储在 {log_parent_fd}/{run_name}，如果不指定，评估结果会存在 project_path.log_fd 下，'
                             '可用 {project_fd} 指代 project_path.project_fd')
    parser.add_argument('-nv', '--no_verbose', dest='verbose', default=True, action='store_false',
                        help='如果指定，则不创建 log 日志，并在计算完 auc 后删除所有评估结果')
    # TODO(islander): 直接读取 pbtxt 文件构建模型
    parser.add_argument('-mo', '--model', default='din', choices=('din', 'lr', 'deepfm', 'wide_deep', 'pnn'),
                        help='训练的机器学习模型')
    parser.add_argument('-bn', '--batch_norm', default=None, choices=(None, 'bn'),
                        help='是否使用 batchnorm')
    parser.add_argument('-ds', '--dataset', default='movielens', choices=('movielens', 'amazon'),
                        help='使用的数据集')
    parser.add_argument('--movie_genome_fp', type=parse_fp,
                        default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'genome-scores.csv'),
                        help='电影的硬编码 embedding 数据的路径')

    args, unparsed_args = parser.parse_known_args()
    if unparsed_args:
        _custom_logger.log_text('WARNING: Found unrecognized sys.argv: {}'.format(unparsed_args))
    return args


def main():
    args = config_args()
    # 文件名参照 task_index 命名，名保证长度一致
    num_len = len(str(args.task_count - 1))  # 最大位宽
    same_len_task_index = '%0{}d'.format(num_len) % args.task_index  # 0 补齐头位，便于 oss 中排序
    log_pfd = osp.join(args.log_parent_fd, args.run_name)  # 记录日志的父目录
    log_fd = osp.join(log_pfd, same_len_task_index)  # 本次运行记录日志的目录，记录在用 task_index 命名的子目录
    gfile.MakeDirs(log_pfd)

    if args.verbose:
        # 创建目录和文件
        gfile.MakeDirs(log_fd)

    metric_names = ['gauc', 'auc', 'accuracy', 'false_prop', 'neg_log_loss', 'square_loss', 'num_samples',
                    'max_true_prob']
    if not args.skip_mapping:
        if args.verbose:
            meta_f = gfile.GFile(osp.join(log_fd, 'meta.txt'), 'a')  # 用于记录一些运行基本信息
        else:
            meta_f = None  # None 则后面因为 custom_logger 的实现，都不会写文件
        try:
            # 记录当前键入的命令
            command = log.get_command()
            _custom_logger.log_text('current command:\n{}'.format(command), file_handler=meta_f)
            # 记录处理后的命令行参数
            args_str = pprint.pformat(args.__dict__)
            _custom_logger.log_text('parsed args:\n' + args_str, file_handler=meta_f)
            if args.verbose:
                # noinspection PyTypeChecker
                json.dump(args.__dict__, fp=gfile.GFile(osp.join(log_fd, 'args.json'), 'w'))  # 将参数记录下来
            _logger.info('命令行参数解析完成')

            # 确定使用的配置
            if args.dataset == 'movielens':
                config_prefix_pkg = config.movielens
            elif args.dataset == 'amazon':
                config_prefix_pkg = config.amazon
            else:
                raise ValueError('Unrecognized dataset {}'.format(args.dataset))
            # 根据模型进一步确认
            if args.model == 'din':
                fea_config = deepcopy(config_prefix_pkg.din.FEA_CONFIG)
                shared_emb_config = deepcopy(config_prefix_pkg.din.SHARED_EMB_CONFIG)
            elif args.model == 'lr':
                fea_config = deepcopy(config_prefix_pkg.lr.FEA_CONFIG)
                shared_emb_config = deepcopy(config_prefix_pkg.lr.SHARED_EMB_CONFIG)
            elif args.model == 'deepfm':
                fea_config = deepcopy(config_prefix_pkg.deepfm.FEA_CONFIG)
                shared_emb_config = deepcopy(config_prefix_pkg.deepfm.SHARED_EMB_CONFIG)
            elif args.model == 'wide_deep':
                fea_config = deepcopy(config_prefix_pkg.wide_deep.FEA_CONFIG)
                shared_emb_config = deepcopy(config_prefix_pkg.wide_deep.SHARED_EMB_CONFIG)
            elif args.model == 'pnn':
                fea_config = deepcopy(config_prefix_pkg.pnn.FEA_CONFIG)
                shared_emb_config = deepcopy(config_prefix_pkg.pnn.SHARED_EMB_CONFIG)
            else:
                raise ValueError('unrecognized args.model = {}'.format(args.model))

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
                net = model.deepfm.DeepFM(input_config=fea_config, shared_emb_config=shared_emb_config,
                                          use_moving_statistics=True)
            elif args.model == 'wide_deep':
                net = model.wide_deep.WideDeep(input_config=fea_config, shared_emb_config=shared_emb_config,
                                               use_moving_statistics=True)
            elif args.model == 'pnn':
                net = model.pnn.PNN(input_config=fea_config, shared_emb_config=shared_emb_config,
                                    use_moving_statistics=True)
            else:
                raise ValueError('unrecognized args.model = {}'.format(args.model))
            _logger.info('初始化模型成功')

            net.build_graph_(key='eval', mode=tf.estimator.ModeKeys.EVAL, device=args.device)
            net.switch_graph('eval')
            net.saver.restore(net.session, args.model_fp)
            _logger.info('模型构建成功，从 {} 加载了参数'.format(args.model_fp))

            if args.verbose:
                log_responsible_fns_f = gfile.GFile(osp.join(log_fd, 'eval_fns.txt'.format(args.task_index)), 'w')
            else:
                log_responsible_fns_f = None
            if args.dataset == 'movielens':
                eval_generator = train_utils.get_movielens_input_fn(
                    data_fd=args.data_fd, mapping_fp=args.mapping_fp, require='generator',
                    fea_config=fea_config, slice_index=args.task_index, slice_count=args.task_count,
                    shuffle=False, log_responsible_fns=log_responsible_fns_f, movie_genome_fp=args.movie_genome_fp)
            elif args.dataset == 'amazon':
                eval_generator = train_utils.get_amazon_input_fn(
                    data_fd=args.data_fd, mapping_fp=args.mapping_fp, require='generator',
                    fea_config=fea_config, slice_index=args.task_index, slice_count=args.task_count,
                    shuffle=False, log_responsible_fns=log_responsible_fns_f, seed_plus=1000000)
            else:
                raise ValueError('Unrecognized dataset {}'.format(args.dataset))
            _logger.info('数据生成器构建完成')

            result_fp = osp.join(log_pfd, 'result_{}.dat'.format(same_len_task_index))  # 记录运算结果，记录在父目录
            with gfile.GFile(result_fp, 'wb') as result_f:
                eval_res = model.evaluate_by_net(net, input_fn=eval_generator, capture_feat_names=('feat/user_id',),
                                                 test_steps=args.num_test_steps, verbose=True)
                eval_res['user_ids'] = eval_res.pop('feat/user_id')  # 重命名 key，便于计算 metric 时一致传参

                for user_id, prob, label in zip(eval_res['user_ids'], eval_res['probs'], eval_res['labels']):
                    result_f.write(struct.pack(PROTOCOL, user_id[0], prob[0], prob[1], label))

                metric_values = log.metrics.get_metrics(eval_res, metrics=metric_names)
                if args.verbose:
                    with gfile.GFile(osp.join(log_fd, 'metrics.csv'), 'w') as metrics_f:
                        metrics_f.write(','.join(metric_names) + '\n')
                        _custom_logger.log_dict(msg=metric_values, hint='partial evaluation result',
                                                file_handler=metrics_f)
                else:
                    _custom_logger.log_dict(msg=metric_values, hint='partial evaluation result')
        finally:
            if meta_f is not None:
                meta_f.close()

    # 进入 reduce，检查文件夹中的文件是否完整，如果完整，进入 reduce，否则说明自己不是最后退出的，进行 reduce
    # 过滤得到文件
    fns = gfile.ListDirectory(log_pfd)
    pat = re.compile(r'result_(\d+)\.dat')
    fns = [fn for fn in fns if re.match(pat, fn)]
    exited_task_indices = set([int(re.match(pat, fn).group(1)) for fn in fns])  # 已经完成的任务索引
    remaining_task_indices = [task_index for task_index in range(args.task_count) if
                              task_index not in exited_task_indices]  # 除去之后得到没有推出的
    is_all_mapper_finished = not bool(remaining_task_indices)  # 剩余为空，说明都退出了，task_count 可以 < 结果数量，发生在使用 --skip_mapping 时

    if is_all_mapper_finished:
        _logger.info('all mappers exited, start reducing')

        user_ids = []
        probs = []
        labels = []
        for fn in fns:
            fp = osp.join(log_pfd, fn)
            with gfile.GFile(fp, 'rb') as result_f:
                print(f'reading evaluation results from {fp}')
                while True:
                    line = result_f.read(NUM_BYTES)
                    if line == b'':
                        break
                    user_id, prob0, prob1, label = struct.unpack(PROTOCOL, line)
                    user_ids.append(user_id)
                    probs.append([prob0, prob1])
                    labels.append(label)

                    if len(user_ids) % 10000 == 0:
                        print(f'{len(user_ids)} sample results read')

        metric_values = log.metrics.get_metrics(
            {'user_ids': np.array(user_ids), 'probs': np.array(probs), 'labels': np.array(labels)},
            metrics=metric_names)
        with gfile.GFile(osp.join(log_pfd, 'metrics.csv'), 'w') as metrics_f:
            metrics_f.write(','.join(metric_names) + '\n')
            _custom_logger.log_dict(msg=metric_values, hint='full evaluation result',
                                    file_handler=metrics_f)
        if not args.verbose:  # 删除文件
            for fn in fns:
                fp = osp.join(log_pfd, fn)
                gfile.Remove(fp)
    else:
        # 算出没有退出的 mapper，并打印日志

        _logger.info('workers with task_index {} have not exited'.format(remaining_task_indices))


if __name__ == '__main__':
    main()
