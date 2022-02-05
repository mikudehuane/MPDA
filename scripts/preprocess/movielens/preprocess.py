# -*- coding: utf-8 -*-
# @Time    : 2021/7/12 下午5:43
# @Author  : islander
# @File    : preprocess.py
# @Software: PyCharm

import _init_paths2
from tensorflow.python.platform import gfile
import csv


def rating_data(data_fp):
    """按用户切分数据的生成器

    Args:
        data_fp: ratings.csv 数据文件的路径，文件第一列用于区分列，允许变化，包含四列：userId, movieId, rating, timestamp

    Yields (Tuple[str, List[Tuple[str, str, str, str]]]):
        按用户生成数据，每一行为 [user_id, movie_id, rating, timestamp]，行顺序与原始顺序相同，每次生成一个元组，user_id, 所有数据的二维数组
    """
    with gfile.GFile(data_fp, 'r') as f:
        heading = f.readline().strip().split(',')
        heading = {name: idx for idx, name in enumerate(heading)}
        user_id_col = heading['userId']
        movie_id_col = heading['movieId']
        rating_col = heading['rating']
        timestamp_col = heading['timestamp']

        cache = []  # 待生成数据的缓存
        for line in f:
            line = line.strip().split(',')
            user_id = line[user_id_col]
            if cache and user_id != cache[0][0]:  # 上一个用户的数据读完了
                yield cache[0][0], cache
                cache.clear()
            line = [user_id, line[movie_id_col], line[rating_col], line[timestamp_col]]
            cache.append(line)
        yield cache[0][0], cache


def main():
    import os.path as osp
    import os
    import project_path

    import argparse
    parser = argparse.ArgumentParser('预处理数据')
    parser.add_argument('--num_users', type=int, default=None, help='指定从数据集里取多少个用户，默认取全部')
    args, _ = parser.parse_known_args()

    input_data_fd = osp.join(project_path.data_fd, 'MovieLens', 'ml-20m')  # 输入文件的目录
    output_data_fd = osp.join(input_data_fd, 'processed')  # 输出文件的目录
    os.makedirs(output_data_fd, exist_ok=True)

    # 按用户分割数据，并按时间戳排序
    for rating_data_idx, (user_id, user_data) in enumerate(rating_data(osp.join(input_data_fd, 'ratings.csv'))):
        if args.num_users is not None and rating_data_idx >= args.num_users:  # 判断是否处理够用户了
            break
        if rating_data_idx % 100 == 0:
            print("processing {}th user's data".format(rating_data_idx))

        user_data.sort(key=lambda x: int(x[3]))  # 按时间戳排序
        with gfile.GFile(osp.join(output_data_fd, '{}.csv'.format(user_id)), 'w') as output_f:
            movie_id_seq = []  # 维护评价过电影的序列
            rating_seq = []  # 维护电影序列对应的评分序列
            for line in user_data:
                user_id, movie_id, rating, timestamp = line
                # 按 ${user_id},${movie_id},${movie_id_seq},${rating_seq},${rating},${timestamp} 构建一行
                line = user_id, movie_id, ' '.join(movie_id_seq), ' '.join(rating_seq), rating, timestamp
                line = ','.join(line) + '\n'
                output_f.write(line)

                # 扩充维护的序列，为下一条数据做准备
                movie_id_seq.append(movie_id)
                rating_seq.append(rating)

    # 类别 id 映射字典
    category_mapping = dict()
    current_category_id = 0

    # 生成电影到类别的映射文件，会根据表头确定列名
    CATEGORY_SPLIT = '|'
    with gfile.GFile(osp.join(input_data_fd, 'movies.csv')) as input_f:
        input_f = csv.reader(input_f)

        heading = input_f.__next__()
        heading = {name: idx for idx, name in enumerate(heading)}
        movie_id_col = heading['movieId']
        category_col = heading['genres']

        with gfile.GFile(osp.join(output_data_fd, 'movie2category.csv'), 'w') as output_f:
            for line in input_f:
                movie_id = line[movie_id_col]
                categories = line[category_col]
                categories = categories.split(CATEGORY_SPLIT)
                for category in categories:  # 添加 id 映射
                    if category not in category_mapping:
                        category_mapping[category] = current_category_id
                        current_category_id += 1
                categories = [str(category_mapping[category]) for category in categories]  # 转成 id
                categories = CATEGORY_SPLIT.join(categories)
                output_f.write(','.join([movie_id, categories]) + '\n')

        with gfile.GFile(osp.join(output_data_fd, 'category.csv'), 'w') as output_f:  # 写类别映射
            output_f = csv.writer(output_f)
            for category_name, category_id in category_mapping.items():
                output_f.writerow([category_id, category_name])


if __name__ == '__main__':
    main()
