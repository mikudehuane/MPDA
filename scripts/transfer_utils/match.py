# -*- coding: utf-8 -*-
# @Time    : 2021/8/6 下午4:44
# @Author  : islander
# @File    : match.py
# @Software: PyCharm
import heapq
import math
import pickle
import random
import time
from copy import deepcopy
import os.path as osp

from tensorflow.python.platform import gfile

import data
from gutils import certain_hash


def get_knn(
        tgt_data, all_data, *, k=None, get_distance, skip_keys
):
    """寻找 k nearest neighbour

    Args:
        - tgt_data: 目标数据，必要
        - all_data (dict): 包含所有数据的字典，其中可能包括 tgt_data，从中找到与目标数据距离最近的 k 条数据，必要
        - k: kNN 的 k，默认为 data_list 的长度
        - get_distance: 计算 tgt_data 和 other_data 之间的距离的函数，由于实现为最小堆，请保证认为更近的值返回值更小
        - skip_keys: all_data 中键在该集合内的被跳过（比如可以指定 tgt_data 所在的 key 来跳过自身）

    Returns:
        一个列表，包含满足 kNN 的 k 条数据在 data_list 中的 index，按照距离从近至远排序。
    """

    class Point(object):  # 用于 最小堆 的内部上浮下沉
        def __init__(self, distance: float = 0.0, _data=None):
            self.distance = distance
            self.data = _data  # 存放 index 而非实际数据

        def __lt__(self, other):  # python 提供了 heapq 最小堆，未提供稳定的最大堆。因此将比较大小的结果反转，1等效于最大堆。
            return self.distance > other.distance  # 取 kNN 时，距离远的会在堆顶，当新的元素 push 入后，将距离最大的堆顶元素 pop

        def __gt__(self, other):
            return self.distance < other.distance

        def __eq__(self, other):
            return self.distance == other.distance

    if k is None:
        k = len(all_data)
    else:
        k = min(k, len(all_data))  # 当 k 大于 len(data_list) 时，将 k 设定为 len(data_list)

    skip_keys = {} if skip_keys is None else skip_keys  # 如果参数为 None 替换为空集合
    min_heap = []
    for key, other_data in all_data.items():
        if key not in skip_keys:  # 跳过指定的数据
            other_distance = get_distance(tgt_data, other_data)
            if len(min_heap) == k:  # 堆已满，pushpop
                heapq.heappushpop(min_heap, Point(distance=other_distance, _data=key))
            else:  # 还有剩余空间，仅 push
                heapq.heappush(min_heap, Point(distance=other_distance, _data=key))

    result_buf = []
    for _ in range(len(min_heap)):
        result_buf.append(heapq.heappop(min_heap).data)  # 因为我们实际模拟的是最大堆，这里 pop 出来的元素实际是先出距离大的，后出距离小的

    result_buf.reverse()  # 将列表反转得到正确从小到大的顺序
    return result_buf


class RandomMatch(object):
    """随机召回

    Args:
        all_user_ids: 被召回的用户的范围
        seed: 随机召回的种子，如果为 None，则直接调用 random.sample，否则创建一个 Random 对象，撒种后调用 sample
        max_match: 召回多少用户
    """
    def __init__(self, all_user_ids, *, seed=None, max_match):
        self._all_user_ids = all_user_ids
        self._max_match = max_match

        if seed is None:
            self._randomer = random
        else:
            self._randomer = random.Random(seed)

    def get_match(self, user_id):
        matched_user_ids = deepcopy(self._all_user_ids)
        matched_user_ids.remove(user_id)  # 删除用户自身
        matched_user_ids = self._randomer.sample(matched_user_ids, self._max_match)
        return matched_user_ids


class MovieIntersectionMatch(object):
    """按照浏览过的电影的交集大小召回

    Args:
        all_user_ids: 被召回的用户的范围
        max_match: 召回多少用户
        data_fd: 存放数据的目录
        verbose: 是否打印进度
        order_by_ratio: 召回排列时，以绝对的交集大小排序还是相对的比例排序
    """
    def __init__(self, all_user_ids, data_fd, *, max_match, verbose=True, order_by_ratio=False):
        # user_id -> 数据中出现过的 movie_id 的集合
        self._movie_id_sets = dict()
        for user_id_idx, user_id in enumerate(all_user_ids):
            self._movie_id_sets[user_id] = set(data.movielens.utils.load_col(
                osp.join(data_fd, '{}.csv'.format(user_id)), 'movie_id'))
            if user_id_idx % 100 == 0 and verbose:
                print('{}/{} users loaded'.format(user_id_idx, len(all_user_ids)))
        if verbose:
            print('{} users loaded into the index'.format(len(self._movie_id_sets)))
        self._max_match = max_match
        self._order_by_ratio = order_by_ratio

    @staticmethod
    def intersect_distance(set1, set2):
        """返回两个集合交集大小的倒数（因为返回的是距离，应该数值越小越接近，所以取倒数）
        """
        intersection_size = len(set.intersection(set1, set2))
        return 1.0 / (intersection_size + 1e-7)

    @staticmethod
    def intersect_distance_ratio(set1, set2):
        """返回 两个集合大小乘积/两个集合交集大小，乘积是为了保证其中一个大小不变时，距离正比于比例
        """
        intersection_size = len(set.intersection(set1, set2))
        average_size = len(set1) * len(set2)
        return average_size / (intersection_size + 1e-7)

    def get_match(self, user_id):
        if self._order_by_ratio:
            get_distance = self.intersect_distance_ratio
        else:
            get_distance = self.intersect_distance

        matched_user_ids = get_knn(
            tgt_data=self._movie_id_sets[user_id], all_data=self._movie_id_sets,
            k=self._max_match, get_distance=get_distance, skip_keys={user_id})
        return matched_user_ids


class MovieIntersectionMultiWorkerMatch(MovieIntersectionMatch):
    """MovieIntersectionMatch 多 worker 协同构建 index 的子类

    Args:
        all_user_ids: 被召回的用户的范围
        max_match: 召回多少用户
        data_fd: 存放数据的目录
        task_index: 当前 worker 的 index
        task_count: 总 worker 数量
        log_fd: 存放日志的目录，用于做缓存索引结果，并通知其他 worker 索引构建完成

    Keyword Args:
        same with MovieIntersectionMatch

    Notes:
        这里用户分配的逻辑为按照哈希值划分，与 worker 负责的用户不一定一致
    """

    def __init__(self, all_user_ids, data_fd, *, task_index, task_count, log_fd, **kwargs):
        self._max_match = kwargs.get('max_match')
        self._task_index = task_index
        self._order_by_ratio = kwargs.get('order_by_ratio', False)
        verbose = kwargs.get('verbose', True)

        def _is_responsible(_user_id):  # 输入 user_id 返回是否为本 worker 负责
            return certain_hash(_user_id) % task_count == task_index

        # 当前 worker 负责的 id
        responsible_user_ids = [user_id for user_id in all_user_ids if _is_responsible(user_id)]
        movie_id_sets_dump_fp = osp.join(log_fd, 'movie_id_sets_{}_{}.pkl'.format(task_index, task_count))
        if not gfile.Exists(movie_id_sets_dump_fp):
            super().__init__(responsible_user_ids, data_fd, **kwargs)
            # noinspection PyTypeChecker
            pickle.dump(self._movie_id_sets, gfile.GFile(movie_id_sets_dump_fp, 'wb'))
        else:
            # noinspection PyTypeChecker
            self._movie_id_sets = pickle.load(gfile.GFile(movie_id_sets_dump_fp, 'rb'))

        # 检查其他 worker 有没有存好
        num_wait_seconds = 1
        while True:
            is_all_worker_terminated = True
            for task_index in range(task_count):
                movie_id_sets_dump_fp = osp.join(log_fd, 'movie_id_sets_{}_{}.pkl'.format(task_index, task_count))
                if not gfile.Exists(movie_id_sets_dump_fp):  # 有 worker 没跑完
                    is_all_worker_terminated = False
                    if verbose:
                        print('worker {} has not terminated, wait {}s for next try'.format(task_index, num_wait_seconds))
                    break
            if is_all_worker_terminated:  # 所有 worker 结束，开始合并 index
                break
            time.sleep(num_wait_seconds)  # 等待一定时间后再次检查

        # 填充其他 worker 的结果
        for task_index in range(task_count):
            movie_id_sets_dump_fp = osp.join(log_fd, 'movie_id_sets_{}_{}.pkl'.format(task_index, task_count))
            if task_index != self._task_index:
                # noinspection PyTypeChecker
                movie_id_sets = pickle.load(gfile.GFile(movie_id_sets_dump_fp, 'rb'))
                self._movie_id_sets.update(movie_id_sets)

                if verbose:
                    print('{} users loaded from {} merged into the index, got {} users'.format(
                        len(movie_id_sets), movie_id_sets_dump_fp, len(self._movie_id_sets)))
