# -*- coding: utf-8 -*-
# @Time    : 2021/8/28 下午12:49
# @Author  : islander
# @File    : statistics.py
# @Software: PyCharm

import csv

import _init_paths2

import json

from tensorflow.python.platform import gfile

import project_path
import os.path as osp
import gutils

data_fd = osp.join(project_path.data_fd, 'amazon', 'Electronics_5')


def main():
    data_fp = osp.join(data_fd, 'rating_only.csv')

    rating_to_count = dict()
    num_samples_to_count = dict()
    user_to_num_samples = dict()
    user_to_num_train_samples = dict()
    user_to_num_test_samples = dict()

    with gfile.GFile(data_fp) as data_f:
        data_f = csv.reader(data_f)
        data_f.__next__()
        for user_id, item_id, rating, timestamp in data_f:
            rating_to_count[rating] = rating_to_count.get(rating, 0) + 1
            user_to_num_samples[user_id] = user_to_num_samples.get(user_id, 0) + 1
            if int(timestamp) < 1385078400:
                user_to_num_train_samples[user_id] = user_to_num_train_samples.get(user_id, 0) + 1
            else:
                user_to_num_test_samples[user_id] = user_to_num_test_samples.get(user_id, 0) + 1

    user_intersection_fp = osp.join(data_fd, 'processed', 'ts=1385078400_user-intersect_full.json')
    user_intersection = json.load(gfile.GFile(user_intersection_fp, 'r'))
    user_intersection_filtered = [user_id for user_id in user_intersection if user_to_num_train_samples[user_id] >= 20]
    # noinspection PyTypeChecker
    # json.dump(user_intersection_filtered, gfile.GFile(osp.join(data_fd, 'processed', 'ts=1385078400_user-intersect_trainset20.json'), 'w'))
    user_intersection_filtered = set(user_intersection_filtered)

    for user_id, num_samples in user_to_num_samples.items():
        num_samples_to_count[num_samples] = num_samples_to_count.get(num_samples, 0) + 1

    for rating, count in rating_to_count.items():
        print('{}: {}'.format(rating, count))

    num_samples_to_count = list(num_samples_to_count.items())
    num_samples_to_count.sort(key=lambda x: int(x[0]))
    print(sum([count for num_samples, count in num_samples_to_count if num_samples >= 20]))
    for num_samples, count in num_samples_to_count:
        print('{}: {}'.format(num_samples, count))

    # 统计训练/测试数据量，只考虑评估的用户
    def _print_num_samples_statistics(_user_id_to_count, _msg):
        _num_samples_to_count = dict()
        for _user_id, _num_samples in _user_id_to_count.items():
            if _user_id in user_intersection_filtered:
                _num_samples_to_count[_num_samples] = _num_samples_to_count.get(_num_samples, 0) + 1

        print(_msg)
        for _num_samples, _count in sorted(list(_num_samples_to_count.items()), key=lambda x: x[0]):
            print('{}: {}'.format(_num_samples, _count))

        print('x <= 10', sum((_num_samples_to_count[num_sample] for num_sample in _num_samples_to_count
                              if num_sample <= 10)))

    _print_num_samples_statistics(user_to_num_train_samples, 'train:')
    _print_num_samples_statistics(user_to_num_test_samples, 'test:')

    # print('20 <= x <= 30:', sum((num_train_samples_to_count[num_sample] for num_sample in num_train_samples_to_count
    #                             if 20 <= num_sample <= 30)))
    # print('20 <= x:', sum((num_train_samples_to_count[num_sample] for num_sample in num_train_samples_to_count
    #                        if num_sample >= 20)))


def category_count():
    category_mapping = osp.join(data_fd, 'processed', 'item2category.csv')
    num_cat2count = dict()
    with gfile.GFile(category_mapping) as category_mapping:
        category_mapping = csv.reader(category_mapping)
        for item, categories in category_mapping:
            categories = categories.split('|')
            num_cat2count[len(categories)] = num_cat2count.get(len(categories), 0) + 1

    num_cat2count = list(num_cat2count.items())
    num_cat2count.sort(key=lambda x: int(x[0]))
    for num_cat, count in num_cat2count:
        print('{}: {}'.format(num_cat, count))


if __name__ == '__main__':
    main()

