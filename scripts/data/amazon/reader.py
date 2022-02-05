# -*- coding: utf-8 -*-
# @Time    : 2021/8/28 下午12:39
# @Author  : islander
# @File    : reader.py
# @Software: PyCharm
import csv
import random
from copy import deepcopy

from tensorflow.python.platform import gfile

import gutils


class DataReader(object):
    """Amazon 数据读取器，接口可以直接接入 model.din.DataGenerator（用于拼接 batch）

    Input data file should obey the following rules:
    - Columns are comma-split, and sequences are space-split
    - Columns order is DataReader.COLUMNS_ORDER

    Args:
        data_fp: 数据文件的路径
        item2categories: 各个商品的类别信息 item_id -> [category_1, category_2, category_3, ...] 类别都是 int，
            item_id 未出现则认为类别是空数组

    Keyword Args:
        rating_thres: 判断为正样本的阈值，大于等于该值被判断为正样本
        config: 特征配置，OrderedDict，输出的顺序与该配置中的键顺序一致，此外输出第一列是标签
    """
    # 数据文件列的顺序
    COLUMNS_ORDER = ['user_id', 'item_id', 'item_id_seq', 'rating_seq', 'rating', 'timestamp']
    COLUMNS_NAME2IDX = {name: idx for idx, name in enumerate(COLUMNS_ORDER)}

    @staticmethod
    def _get_col(row, col_name):  # 获取一行数据的指定列
        return row[DataReader.COLUMNS_NAME2IDX[col_name]]

    def __init__(self, data_fp, item2categories, *, config,
                 rating_thres=5.0):
        self.data_fp = data_fp
        self.fea_config = config
        self.item2categories = item2categories
        self.rating_thres = rating_thres

        self._split_his = ' '

        # 创建 reader，如果文件不存在，创建一个空数据集
        if gfile.Exists(data_fp):
            self._reader = gfile.GFile(data_fp, 'r')
            self._csv_reader = csv.reader(self._reader)
        else:
            self._reader = None
            self._csv_reader = None

    def read(self, size=1):
        """

        Args:
            size: 给定读多少航

        Returns: A 2d array
            - 每行一个样本
            - 行顺序为 标签，输入（顺序与 self.fea_config 一致）
            - 历史评价序列联合处理 item_id_seq 和 rating_seq 得到
            - 类别数量不确定，最多取多少个类别，以及填充不足的类别逻辑，由拼 batch 的 DataGenerator 完成
        """
        if self.is_no_data_left():
            raise EOFError('Reader reaches EOF.')

        ret = []
        for _ in range(size):
            try:
                raw_row = self._csv_reader.__next__()
            except StopIteration:  # 数据读完了，退出循环
                break

            # 取一行数据到变量
            user_id = int(self._get_col(raw_row, 'user_id'))
            item_id = int(self._get_col(raw_row, 'item_id'))
            # 如果是空字符串，为空序列
            item_id_seq = self._get_col(raw_row, 'item_id_seq')
            item_id_seq = [int(x) for x in item_id_seq.split(' ')] if item_id_seq else []
            # 如果是空字符串，为空序列
            rating_seq = self._get_col(raw_row, 'rating_seq')
            rating_seq = [float(x) for x in rating_seq.split(' ')] if rating_seq else []
            rating = float(self._get_col(raw_row, 'rating'))

            # 获取标签
            label = 1 if gutils.robust_geq(rating, self.rating_thres) else 0

            # 解析非序列全部特征
            features = dict()
            features['user_id'] = user_id
            features['item_id'] = item_id
            features['category_ids'] = self.item2categories.get(item_id, [])

            # 解析序列特征
            features['item_id_seq_pos'] = []
            features['category_ids_seq_pos'] = []
            features['item_id_seq_neg'] = []
            features['category_ids_seq_neg'] = []
            for item_id, rating in zip(item_id_seq, rating_seq):
                category_ids = self.item2categories.get(item_id, [])
                if gutils.robust_geq(rating, self.rating_thres):
                    features['item_id_seq_pos'].append(item_id)
                    features['category_ids_seq_pos'].append(category_ids)
                else:
                    features['item_id_seq_neg'].append(item_id)
                    features['category_ids_seq_neg'].append(category_ids)

            # 按顺序填充到一行，仅填充配置中出现的特征，丢弃其他特征
            row = [label]
            for key in self.fea_config:
                row.append(features[key])

            # 将该行内容添加到返回值
            ret.append(row)

        return ret

    def seek(self, offset=0):
        assert offset == 0  # 只支持回到文件开头
        if self._reader is not None:
            self._reader.seek(0)

    def close(self):
        if self._reader is not None:
            self._reader.close()

    def is_no_data_left(self):
        if self._reader is None:
            return True
        else:
            return self._reader.tell() == self._reader.size()


class DataReaderNegSampling(object):
    """Amazon 数据读取器，正样本表示出现在数据集，负样本通过 1：1 负采样获得

    Input data file should obey the following rules:
    - Columns are comma-split, and sequences are space-split
    - Columns order is DataReader.COLUMNS_ORDER

    Args:
        data_fp: 数据文件的路径
        item2categories: 各个商品的类别信息 item_id -> [category_1, category_2, category_3, ...] 类别都是 int，
            item_id 未出现则认为类别是空数组

    Keyword Args:
        num_items: item 的数量，用于确定负采样的范围
        config: 特征配置，OrderedDict，输出的顺序与该配置中的键顺序一致，此外输出第一列是标签
        seed: 负采样的随机数种子，必须给定，以保证数据集不具有随机性
    """
    # 数据文件列的顺序
    COLUMNS_ORDER = ['user_id', 'item_id', 'item_id_seq', 'rating_seq', 'rating', 'timestamp']
    COLUMNS_NAME2IDX = {name: idx for idx, name in enumerate(COLUMNS_ORDER)}

    @staticmethod
    def _get_col(row, col_name):  # 获取一行数据的指定列
        return row[DataReader.COLUMNS_NAME2IDX[col_name]]

    def __init__(self, data_fp, item2categories, *, num_items=63001, config, seed):
        self.data_fp = data_fp
        self.fea_config = config
        self.item2categories = item2categories
        self._seed = seed
        self._randomer = random.Random(self._seed)

        self._num_items = num_items

        self._split_his = ' '

        # 创建 reader，如果文件不存在，创建一个空数据集
        if gfile.Exists(data_fp):
            self._reader = gfile.GFile(data_fp, 'r')
            self._csv_reader = csv.reader(self._reader)
        else:
            self._reader = None
            self._csv_reader = None

    def read(self, size=1):
        """

        Args:
            size: 给定读多少航

        Returns: A 2d array
            - 每行一个样本
            - 行顺序为 标签，输入（顺序与 self.fea_config 一致）
            - 历史评价序列联合处理 item_id_seq 和 rating_seq 得到
            - 类别数量不确定，最多取多少个类别，以及填充不足的类别逻辑，由拼 batch 的 DataGenerator 完成
        """
        if size % 2 != 0:
            raise ValueError('DataReaderNegSampling does not currently allow odd size, but got {}'.format(size))

        if self.is_no_data_left():
            raise EOFError('Reader reaches EOF.')

        ret = []
        for _ in range(size // 2):
            try:
                raw_row = self._csv_reader.__next__()
            except StopIteration:  # 数据读完了，退出循环
                break

            # 取一行数据到变量
            user_id = int(self._get_col(raw_row, 'user_id'))
            item_id = int(self._get_col(raw_row, 'item_id'))
            # 如果是空字符串，为空序列
            item_id_seq = self._get_col(raw_row, 'item_id_seq')
            item_id_seq = [int(x) for x in item_id_seq.split(' ')] if item_id_seq else []

            # 解析非序列全部特征
            features = dict()
            features['user_id'] = user_id
            features['item_id'] = item_id
            features['category_ids'] = self.item2categories.get(item_id, [])

            # 解析序列特征
            features['item_id_seq'] = []
            features['category_ids_seq'] = []
            for item_id in item_id_seq:
                category_ids = self.item2categories.get(item_id, [])
                features['item_id_seq'].append(item_id)
                features['category_ids_seq'].append(category_ids)

            # 正常填充正样本
            # 按顺序填充到一行，仅填充配置中出现的特征，丢弃其他特征
            row = [1]
            for key in self.fea_config:
                row.append(features[key])
            # 将该行内容添加到返回值
            ret.append(row)

            # 负采样
            features = deepcopy(features)  # 不共用内存，避免出现很难发现的 bug
            sampled_item_id = item_id
            while sampled_item_id == item_id:
                # amazon 数据集处理中，ID 从 1 开始
                sampled_item_id = self._randomer.randint(1, self._num_items)
            features['item_id'] = sampled_item_id
            features['category_ids'] = self.item2categories.get(sampled_item_id, [])
            row = [0]
            for key in self.fea_config:
                row.append(features[key])
            # 将该行内容添加到返回值
            ret.append(row)

        return ret

    def seek(self, offset=0):
        assert offset == 0  # 只支持回到文件开头
        self._randomer = random.Random(self._seed)  # 重置随机数种子，多次读数据数据相同
        if self._reader is not None:
            self._reader.seek(0)

    def close(self):
        if self._reader is not None:
            self._reader.close()

    def is_no_data_left(self):
        if self._reader is None:
            return True
        else:
            return self._reader.tell() == self._reader.size()


class DataReaderMultiple(object):
    """读取多个用户的数据

    Args:
        data_fps: 数据文件路径的列表
        item2categories: 各个电影的类别信息 item_id -> [category_1, category_2, category_3, ...] 类别都是 int
        kernel_reader_class: 内核的数据读取类
        kwargs_list: 字典的列表，与 data_fps 一一对应，用于给每个 reader 传入不同的参数,
            键是传入 kernel_reader_class 初始化的关键字

    Keyword Args: 传入 kernel_reader_class.__init__ 的参数
    """
    def __init__(self, data_fps, item2categories, kernel_reader_class=DataReader, kwargs_list=None, **kwargs):
        self.data_fps = data_fps
        self.item2categories = item2categories
        self.kernel_reader_class = kernel_reader_class
        self._kwargs = kwargs
        self._kwargs_list = kwargs_list

        if data_fps:
            self._set_current_file(0)  # 初始化 self._reader 为第一个文件的读取器

    def is_empty(self):  # 本数据集是否给定了空的文件列表（空数据集但是给了文件列表返回 False）
        return len(self.data_fps) == 0

    def _set_current_file(self, offset):  # 设定当前的文件，并创建读取器
        self._current_data_fp_offset = offset
        if self._kwargs_list is None:
            reader_specific_kwargs = dict()
        else:
            reader_specific_kwargs = self._kwargs_list[offset]
        self._reader = self.kernel_reader_class(data_fp=self.data_fps[self._current_data_fp_offset],
                                                item2categories=self.item2categories, **reader_specific_kwargs, **self._kwargs)

    def _move_to_next_file(self):  # 开始读下一个文件
        assert not self.is_empty()  # 空文件列表不应该执行到这个函数
        self._set_current_file(self._current_data_fp_offset + 1)

    def _move_to_next_non_empty_file(self):
        """移动到当前或下一个有剩余数据的文件，或最后一个文件

        Returns (bool): 移动后是否还有数据剩余
        """
        assert not self.is_empty()  # 空文件列表不应该执行到这个函数
        while self._reader.is_no_data_left():  # 当前 reader 没有数据剩余
            if self._current_data_fp_offset == len(self.data_fps) - 1:  # 已经是最后一个文件了
                return False  # 已经到了最后一个文件，不剩数据了
            else:
                self._move_to_next_file()
        return True  # 移动后还有数据剩余

    def is_no_data_left(self):
        if self.is_empty():
            return True

        is_data_left = self._move_to_next_non_empty_file()
        return not is_data_left

    def read(self, size=1):
        if self.is_no_data_left():  # 该函数一般还会将 self._reader 指针移动到最近一个非空用户
            raise EOFError('all user data reaches EOF')

        ret = []
        while len(ret) < size:  # 读数据直到读到足够的量
            is_data_left = self._move_to_next_non_empty_file()  # 开始读下一个非空文件
            if not is_data_left:  # 所有文件都读完了，退出
                break

            res_size = size - len(ret)  # 还需要多少条数据
            records = self._reader.read(res_size)  # 尝试从当前的 reader 读这么多数据，这里读的数据量可能比 res_size 少
            ret.extend(records)

        return ret

    def seek(self, offset=0):
        """移动 reader 的指针

        Args:
            offset: 只接受 0，重置 reader
        """
        assert offset == 0

        if not self.is_empty():
            self._set_current_file(0)  # 重置文件
            self._reader.seek(0)  # 重置行号
