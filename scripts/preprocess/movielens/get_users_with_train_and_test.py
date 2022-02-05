# -*- coding: utf-8 -*-
# @Time    : 2021/7/28 上午10:40
# @Author  : islander
# @File    : get_users_with_train_and_test.py
# @Software: PyCharm

"""记录下同时有训练和测试数据的用户的列表
"""
import json

import _init_paths2
from tensorflow.python.platform import gfile

import project_path
from gutils import parse_fp, get_user_ids
import os.path as osp


def config_args():
    import argparse

    parser = argparse.ArgumentParser(description='记录下同时有训练和测试数据的用户的列表，用 json 格式记录，按 int(user_id) 排序')
    parser.add_argument('-tfd', '--train_data_fd', type=parse_fp,
                        default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed', 'ts=1225642324_train'),
                        help='训练数据目录的绝对路径')
    parser.add_argument('-tefd', '--test_data_fd', type=parse_fp,
                        default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed', 'ts=1225642324_test'),
                        help='测试数据目录的绝对路径')
    parser.add_argument('-ofp', '--output_fp', type=parse_fp,
                        default=osp.join(project_path.data_fd, 'MovieLens', 'ml-20m', 'processed', 'ts=1225642324_user-intersect.json'))

    args, unparsed_args = parser.parse_known_args()

    if unparsed_args:
        print('WARNING: Found unrecognized sys.argv: {}'.format(unparsed_args))

    return args


def main():
    args = config_args()
    train_user_ids = get_user_ids([args.train_data_fd], require='set')
    test_user_ids = get_user_ids([args.test_data_fd], require='set')

    intersect_user_ids = list(set.intersection(train_user_ids, test_user_ids))
    intersect_user_ids.sort(key=lambda x: int(x))
    # noinspection PyTypeChecker
    json.dump(intersect_user_ids, gfile.GFile(args.output_fp, 'w'))


if __name__ == '__main__':
    main()
