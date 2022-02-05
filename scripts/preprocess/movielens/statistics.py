# -*- coding: utf-8 -*-
# @Time    : 2021/7/15 下午4:22
# @Author  : islander
# @File    : statistics.py
# @Software: PyCharm

"""统计数据集中的用户数量等信息
"""


def main():
    import os.path as osp
    import project_path
    from tensorflow.python.platform import gfile
    import csv

    input_data_fd = osp.join(project_path.data_fd, 'ml-20m')  # 输入文件的目录

    CATEGORY_SPLIT = '|'
    with gfile.GFile(osp.join(input_data_fd, 'movies.csv')) as input_f:
        input_f = csv.reader(input_f)

        heading = input_f.__next__()
        heading = {name: idx for idx, name in enumerate(heading)}
        movie_id_col = heading['movieId']
        category_col = heading['genres']

        category_num2count = dict()  # 类别数到计数
        movie_ids = set()
        for line in input_f:
            movie_id = line[movie_id_col]
            categories = line[category_col]
            categories = categories.split(CATEGORY_SPLIT)
            num_categories = len(categories)
            category_num2count[num_categories] = category_num2count.get(num_categories, 0) + 1
            movie_ids.add(int(movie_id))
        print('类别数量到电影数的字典：'.format(category_num2count))  # 1: 10829, 2: 8809, 3: 5330, 4: 1724, 5: 477, 6: 83, 7: 20, 8: 5, 10: 1
        print('电影 ID 最小值：{}，最大值：{}，数量：{}'.format(min(movie_ids), max(movie_ids), len(movie_ids)))  # 1, 131262, 27278

    user_ids = set()
    with gfile.GFile(osp.join(input_data_fd, 'ratings.csv'), 'r') as f:
        heading = f.readline().strip().split(',')
        heading = {name: idx for idx, name in enumerate(heading)}
        user_id_col = heading['userId']

        for line in f:
            line = line.strip().split(',')
            user_id = line[user_id_col]
            user_ids.add(int(user_id))
    print('user_id 最小是：{}，最大值: {}，数量：{}'.format(min(user_ids), max(user_ids), len(user_ids)))  # 1, 138493, 138493


if __name__ == '__main__':
    main()
