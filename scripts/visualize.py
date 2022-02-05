# -*- coding: utf-8 -*-
# @Time    : 2022/1/18 15:23
# @Author  : islander
# @File    : visualize.py
# @Software: PyCharm
import ast
import csv
import json
import os

import numpy as np
import pandas as pd

import config.movielens.din
import data
import project_path
import os.path as osp
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import re


def get_par(ckpt_path, variable_name):
    """从检查点读参数

    Args:
        ckpt_path: 检查点路径
        variable_name: 变量名

    Notes:
        embedding/user_id
        embedding/category_emb
        embedding/movie_emb

    Returns (np.ndarray):
        参数值
    """
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    return reader.get_tensor(variable_name)


def load_source_user_ids(path, as_int=True):
    """从文件加载源用户的 ID

    Args:
        path: meta.txt 文件
        as_int: 是否以 int 返回 ID
    """
    pat = re.compile('matched users:\n(\\[.*?])')
    user_ids = re.search(pat, open(path).read()).group(1)
    user_ids = ast.literal_eval(user_ids)
    if as_int:
        user_ids = [int(x) for x in user_ids]
    return user_ids


def main():

    def _get_movies(_user_id):
        # 返回用户好评的电影和差评的电影 set, set
        # category 所在的 index，0 处为 label，因此 +1
        _label_index = 0
        _movie_id_index = list(fea_config.keys()).index('movie_id') + 1
        _data_reader = data.movielens.DataReader(
            osp.join(data_fd, '{}.csv'.format(_user_id)),
            movie2categories=movie2categories, config=fea_config)
        _pos_movie_ids = set()
        _neg_movie_ids = set()
        while not _data_reader.is_no_data_left():
            _row = _data_reader.read(1)[0]
            _movie_id = _row[_movie_id_index]
            _label = _row[0]
            if _label:  # 仅考虑正样本
                _pos_movie_ids.add(_movie_id)
            else:
                _neg_movie_ids.add(_movie_id)
        return _pos_movie_ids, _neg_movie_ids

    vis_pfd = osp.join(project_path.log_fd, 'visualize')
    data_fd_raw = osp.join(project_path.data_fd, 'MovieLens', 'ml-20m')
    data_fd = osp.join(data_fd_raw, 'processed')
    checkpoint_fp = '/root/Data-Sharing-Transfer/cloud_models/movielens_din/checkpoint/epoch/model.ckpt'
    os.makedirs(vis_pfd, exist_ok=True)

    run_fd = osp.join(project_path.log_fd, 'transfer_m-200_lr0.01')
    user_ids = os.listdir(run_fd)
    pat = re.compile(r'\d+')
    user_ids = [fn for fn in user_ids if re.match(pat, fn)]

    movie2categories = data.movielens.utils.load_category_mapping(osp.join(data_fd, 'movie2category.csv'))

    fea_config = config.movielens.din.FEA_CONFIG

    # 读取电影元数据，id -> (name, category)
    movies_meta = pd.read_csv(osp.join(data_fd_raw, 'movies.csv'))
    movies_meta = dict(zip(list(movies_meta.movieId),
                           zip(list(movies_meta.title), list(movies_meta.genres))))

    sess = tf.Session()
    full_movie_emb_data = get_par(checkpoint_fp, 'embedding/movie_emb')
    for target_user_id in user_ids[:3]:
        target_fd = osp.join(run_fd, target_user_id)
        source_user_ids = load_source_user_ids(osp.join(target_fd, 'meta.txt'))
        selected_user_ids = set(int(user_id) for user_id in
                                json.load(open(osp.join(target_fd, 'selected_user_ids.json'))))

        with tf.variable_scope(target_user_id):
            target_movies_pos, target_movies_neg = _get_movies(target_user_id)
            # 将外部用户的电影收集到一个集合
            selected_movies_pos, selected_movies_neg = set(), set()
            discarded_movies_pos, discarded_movies_neg = set(), set()
            for source_user_id in source_user_ids:
                if source_user_id in selected_user_ids:
                    processing_pos, processing_neg = selected_movies_pos, selected_movies_neg
                else:
                    processing_pos, processing_neg = discarded_movies_pos, discarded_movies_neg
                current_pos, current_neg = _get_movies(source_user_id)
                processing_pos.update(current_pos)
                processing_neg.update(current_neg)

            # 所有可视化的电影
            involved_movies = set.union(target_movies_pos, target_movies_neg,
                                        selected_movies_pos, selected_movies_neg,
                                        discarded_movies_pos, discarded_movies_neg)
            # 保证顺序
            target_movies_pos_li = list(target_movies_pos)
            selected_movies_pos_li = list(selected_movies_pos)
            other_movies_li = [mid for mid in involved_movies
                               if mid not in target_movies_pos and mid not in selected_movies_pos]

            # 注意 target 与 selected 点有可能重合
            target_movies_emb = [full_movie_emb_data[mid] for mid in target_movies_pos_li]
            selected_movies_emb = [full_movie_emb_data[mid] for mid in selected_movies_pos_li]
            other_movies_emb = [full_movie_emb_data[mid] for mid in other_movies_li]
            visualize_data = np.concatenate([target_movies_emb, selected_movies_emb, other_movies_emb])

            with open(osp.join(vis_pfd, 'u{}_movie.tsv'.format(target_user_id)), 'w') as f:
                f.write('movie\tcategory\tresult\n')
                [f.write('{}\t{}\ttarget\n'.format(*movies_meta[mid]))
                 for mid in target_movies_pos_li]
                [f.write('{}\t{}\tselected\n'.format(*movies_meta[mid]))
                 for mid in selected_movies_pos_li]
                [f.write('{}\t{}\tother\n'.format(*movies_meta[mid]))
                 for mid in other_movies_li]
            # 可视化 category ID，将类别 ID 加载，按用户计数（求均值）
            movie_emb_var = tf.get_variable('movie_emb'.format(target_user_id),
                                            shape=visualize_data.shape)
            movie_emb_var.load(visualize_data, sess)
            print('user {} visualized'.format(target_user_id))

    # 存图可视化
    saver = tf.train.Saver(max_to_keep=None)
    saver.save(sess, osp.join(vis_pfd, 'data.ckpt'))
    summary_writer = tf.summary.FileWriter(vis_pfd)
    summary_writer.add_graph(sess.graph)


if __name__ == '__main__':
    main()
