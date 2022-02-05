# -*- coding: utf-8 -*-
# @Time    : 2021/7/15 下午3:13
# @Author  : islander
# @File    : reader.py
# @Software: PyCharm

"""MovieLens 输入的 csv 数据读取
"""
from collections import OrderedDict

from tensorflow.python.platform import gfile
import csv
import numpy as np


def _robust_geq(larg, rarg, eps=1e-7):
    """算数鲁邦的 >=，当 larg > rarg-eps 时，即返回 True
    """
    return larg > rarg - eps


class DataReader(object):
    """MovieLens 数据读取器，接口可以直接接入 model.din.DataGenerator（用于拼接 batch）

    Input data file should obey the following rules:
    - Columns are comma-split, and sequences are space-split
    - Columns order is DataReader.COLUMNS_ORDER

    Args:
        data_fp: 数据文件的路径
        movie2categories: 各个电影的类别信息 movie_id -> [category_1, category_2, category_3, ...] 类别都是 int，
            movie_id 未出现则认为类别是空数组

    Keyword Args:
        rating_thres: 判断为正样本的阈值，大于等于该值被判断为正样本
        config: 特征配置，OrderedDict，输出的顺序与该配置中的键顺序一致，此外输出第一列是标签
        movie_genomes: 电影的"基因"数据字典，可能有 id 缺失，缺失的 id 用字典中全部数据的均值代替

    Notes:
        For the sample dataset, meta information is as follows:
            max user id: 49022
            number of user: 49022
            max good id: 143533
            number of good: 143533
            max category id: 4814
            number of categories: 4814
    """
    # 数据文件列的顺序
    COLUMNS_ORDER = ['user_id', 'movie_id', 'movie_id_seq', 'rating_seq', 'rating', 'timestamp']
    COLUMNS_NAME2IDX = {name: idx for idx, name in enumerate(COLUMNS_ORDER)}

    @staticmethod
    def _get_col(row, col_name):  # 获取一行数据的指定列
        return row[DataReader.COLUMNS_NAME2IDX[col_name]]

    def __init__(self, data_fp, movie2categories, *, config,
                 rating_thres=4.0, movie_genomes=None):
        self.data_fp = data_fp
        self.fea_config = config
        self.movie2categories = movie2categories
        self.rating_thres = rating_thres
        self._movie_genomes = movie_genomes

        # 求电影基因组的均值向量，用于填充找不到 ID 的电影
        if self._movie_genomes is not None:
            movie_genomes_arr = list(self._movie_genomes.values())
            self._movie_genomes_mean = np.mean(movie_genomes_arr, axis=0)
            del movie_genomes_arr
        else:
            self._movie_genomes_mean = None

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
            - 历史评价序列联合处理 movie_id_seq 和 rating_seq 得到
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
            movie_id = int(self._get_col(raw_row, 'movie_id'))
            # 如果是空字符串，为空序列
            movie_id_seq = self._get_col(raw_row, 'movie_id_seq')
            movie_id_seq = [int(x) for x in movie_id_seq.split(' ')] if movie_id_seq else []
            # 如果是空字符串，为空序列
            rating_seq = self._get_col(raw_row, 'rating_seq')
            rating_seq = [float(x) for x in rating_seq.split(' ')] if rating_seq else []
            rating = float(self._get_col(raw_row, 'rating'))

            # 获取标签
            label = 1 if _robust_geq(rating, self.rating_thres) else 0

            # 解析非序列全部特征
            features = dict()
            features['user_id'] = user_id
            features['movie_id'] = movie_id
            features['category_ids'] = self.movie2categories.get(movie_id, [])

            # 解析序列特征
            features['movie_id_seq_pos'] = []
            features['category_ids_seq_pos'] = []
            features['movie_id_seq_neg']= []
            features['category_ids_seq_neg'] = []
            for movie_id, rating in zip(movie_id_seq, rating_seq):
                category_ids = self.movie2categories[movie_id]
                if _robust_geq(rating, self.rating_thres):
                    features['movie_id_seq_pos'].append(movie_id)
                    features['category_ids_seq_pos'].append(category_ids)
                else:
                    features['movie_id_seq_neg'].append(movie_id)
                    features['category_ids_seq_neg'].append(category_ids)

            # 按顺序填充到一行，仅填充配置中出现的特征，丢弃其他特征
            row = [label]
            for key in self.fea_config:
                if key == 'movie_genome':
                    movie_genome = self._movie_genomes.get(movie_id, self._movie_genomes_mean)

                    row.append(movie_genome)
                else:
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


class DataReaderMultiple(object):
    """读取多个用户的数据

    Args:
        data_fps: 数据文件路径的列表
        movie2categories: 各个电影的类别信息 movie_id -> [category_1, category_2, category_3, ...] 类别都是 int

    Keyword Args: 传入 DataReader 的参数
    """
    def __init__(self, data_fps, movie2categories, **kwargs):
        self.data_fps = data_fps
        self.movie2categories = movie2categories
        self._kwargs = kwargs

        if data_fps:
            self._set_current_file(0)  # 初始化 self._reader 为第一个文件的读取器

    def is_empty(self):  # 本数据集是否给定了空的文件列表（空数据集但是给了文件列表返回 False）
        return len(self.data_fps) == 0

    def _set_current_file(self, offset):  # 设定当前的文件，并创建读取器
        self._current_data_fp_offset = offset
        self._reader = DataReader(data_fp=self.data_fps[self._current_data_fp_offset],
                                  movie2categories=self.movie2categories, **self._kwargs)

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
