# -*- coding: utf-8 -*-
# @Time    : 2021/7/16 上午10:54
# @Author  : islander
# @File    : utils.py
# @Software: PyCharm

from tensorflow.python.platform import gfile
import csv
import pandas as pd


def load_category_mapping(input_fp):
    """读取电影到类别的映射

    Args:
        input_fp: 映射文件的路径

    Returns (Dict[int, List[int]]):
        movie_id -> [category_id1, category_id2, ...]
    """
    with gfile.GFile(input_fp) as input_f:
        input_f = csv.reader(input_f)
        ret = dict()
        for movie_id, category_ids in input_f:
            movie_id = int(movie_id)
            category_ids = [int(category_id) for category_id in category_ids.split('|')]
            ret[movie_id] = category_ids
        return ret


def load_genome(input_fp, verbose=True):
    """加载电影的"基因数据"，即与各个 tag 的关联性

    Args:
        verbose: 是否打印进度
        input_fp: 数据文件的路径

    Returns:
        Dict[int, List[float]]: 电影 ID 向"基因"映射的字典
    """
    NUM_TAGS = 1128

    with gfile.GFile(input_fp) as input_f:
        input_f = csv.reader(input_f)
        input_f.__next__()  # 跳过表头

        def movie_genomes():  # 按 movie 生成数据
            ret = []
            prev_movie_id = None
            for line in input_f:
                _movie_id = line[0]
                if _movie_id != prev_movie_id and prev_movie_id is not None:  # movie 改变了
                    assert len(ret) == NUM_TAGS
                    yield prev_movie_id, ret
                    ret = []
                ret.append(line)
                prev_movie_id = _movie_id

        mid_to_genome = dict()
        for movie_idx, (movie_id, lines) in enumerate(movie_genomes()):
            prev_tag_id = 0
            genome = []
            for _, tag_id, relevance in lines:
                tag_id = int(tag_id)
                relevance = float(relevance)
                assert tag_id == prev_tag_id + 1  # 确保 tag 是顺序记录的
                genome.append(relevance)
                prev_tag_id = tag_id
            mid_to_genome[int(movie_id)] = genome

            if movie_idx % 100 == 0:
                if verbose:
                    print('{} movies processed'.format(movie_idx + 1))

    return mid_to_genome


def load_col(input_fp, col_name):
    """读取一列数据

    Args:
        col_name: 希望读取的列名，取 ['user_id', 'movie_id', 'movie_id_seq', 'rating_seq', 'rating', 'timestamp'] 之一
        input_fp: 数据文件的路径

    Returns:
        list[str]: 数据列的列表
    """
    # noinspection PyTypeChecker
    input_data = pd.read_csv(
        gfile.GFile(input_fp), dtype=str, header=None, usecols=[col_name],
        names=['user_id', 'movie_id', 'movie_id_seq', 'rating_seq', 'rating', 'timestamp'])
    col_data = list(input_data[col_name].values)
    return col_data
