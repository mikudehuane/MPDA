# -*- coding: utf-8 -*-
# @Time    : 2021/7/13 下午5:40
# @Author  : islander
# @File    : split.py
# @Software: PyCharm

import _init_paths2
import re
import project_path
import os.path as osp
from tensorflow.python.platform import gfile
import csv


def config_args():
    import argparse
    import sys
    from gutils import parse_fp

    unparsed_args = sys.argv[1:]

    parser0 = argparse.ArgumentParser(add_help=False)
    parser0.add_argument('-ifd', '--input_data_fd', type=parse_fp,
                         default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed'),
                         help='待处理数据目录')
    parser0.add_argument('-sm', '--split_method', type=str, choices=('timestamp', 'user'), default='timestamp',
                         help='切分数据集的方法，按时间戳切分或者按用户切分')
    args, unparsed_args = parser0.parse_known_args(args=unparsed_args)

    parser1 = argparse.ArgumentParser(add_help=False)
    if args.split_method == 'timestamp':
        parser1.add_argument('-ts', '--timestamp', type=str, default='1225642324',
                             help='按时间戳分割训练集合测试集，小于该时间戳的数据分到训练，大于等于的分到测试')
    else:
        parser1.add_argument('-ur', '--eval_user_ratio', type=float, default=0.25,
                             help='按日期划分，用户的比例')
    args, unparsed_args = parser1.parse_known_args(args=unparsed_args, namespace=args)

    if unparsed_args:
        print('WARNING: Found unrecognized sys.argv: {}'.format(unparsed_args))

    parser_help = argparse.ArgumentParser(description='分割数据为训练集和测试集', parents=[parser0, parser1])
    parser_help.parse_known_args()

    return args


def main():

    args = config_args()

    data_fn_pat = re.compile(r'\d+\.csv')  # 用户数据的文件名
    input_fns = gfile.ListDirectory(args.input_data_fd)
    input_fns = [fn for fn in input_fns if re.match(data_fn_pat, fn)]  # 过滤数据文件，跳过映射文件等其他文件

    # 训练集和测试集存放的目录
    if args.split_method == 'timestamp':
        train_fd = osp.join(args.input_data_fd, 'ts={}_train'.format(args.timestamp))
        test_fd = osp.join(args.input_data_fd, 'ts={}_test'.format(args.timestamp))
    else:
        train_fd = osp.join(args.input_data_fd, 'ur={}_train'.format(args.eval_user_ratio))
        test_fd = osp.join(args.input_data_fd, 'ur={}_test'.format(args.eval_user_ratio))
    gfile.MakeDirs(train_fd)
    gfile.MakeDirs(test_fd)

    if args.split_method == 'timestamp':
        for idx, input_fn in enumerate(input_fns):
            # 打开输入和输出文件，注意 GFile 不同于 open，初始化时不创建文件，真正写入内容才创建
            input_f = gfile.GFile(osp.join(args.input_data_fd, input_fn), 'r')
            output_train_f = gfile.GFile(osp.join(train_fd, input_fn), 'w')
            output_test_f = gfile.GFile(osp.join(test_fd, input_fn), 'w')

            try:
                input_f_csv = csv.reader(input_f)
                output_train_f_csv = csv.writer(output_train_f)
                output_test_f_csv = csv.writer(output_test_f)
                for line in input_f_csv:
                    line_ts = line[-1]  # 最后一列是时间戳
                    if int(line_ts) < int(args.timestamp):  # 用整数比较
                        output_train_f_csv.writerow(line)
                    else:
                        output_test_f_csv.writerow(line)
            finally:
                input_f.close()
                output_train_f.close()
                output_test_f.close()
            if idx % 100 == 0:
                print('{} users processed'.format(idx))
    elif args.split_method == 'user':  # 按用户划分
        for idx, input_fn in enumerate(input_fns):
            # 通过哈希值来划分训练和测试，python hash 函数每次运行返回结果可能不同，因此每次的划分有一定随机性
            hash_max = 100
            input_fn_hash = hash(input_fn) % hash_max
            train_hash_max = int(hash_max * args.eval_user_ratio)

            input_fp = osp.join(args.input_data_fd, input_fn)
            if input_fn_hash < train_hash_max:
                output_fp = osp.join(train_fd, input_fn)
            else:
                output_fp = osp.join(test_fd, input_fn)

            # 这样写比直接调用 gfile.copy 快
            gfile.GFile(output_fp, 'w').write(gfile.GFile(input_fp).read())

            if idx % 100 == 0:
                print('{} users processed'.format(idx))
    else:
        raise ValueError('unrecognized split_method: {}'.format(args.split_method))


if __name__ == '__main__':
    main()
