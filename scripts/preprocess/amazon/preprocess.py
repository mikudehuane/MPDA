# -*- coding: utf-8 -*-
# @Time    : 2021/8/27 下午12:19
# @Author  : islander
# @File    : preprocess.py
# @Software: PyCharm
import _init_paths2
import csv
import json
import project_path
import os.path as osp

from tensorflow.python.platform import gfile


def extract_data(input_fp, output_fp, mapping_fp):
    """从原始数据中提取需要的项目，写为 csv 文件
    csv 文件每一行为一条数据，按顺序包含信息：
    - user_id（从 1 开始）
    - item_id（从 1 开始）
    - rating（评分）
    - timestamp（时间戳，int 类型，与数据源 unixReviewTime 一致）

    Args:
        input_fp: 原始数据的路径
        output_fp: 输出文件的路径
        mapping_fp: 输出 ID 映射信息的文件路径
    """
    def _get_id(_id_dict, _raw_id, _start_id=1):
        """根据字典 _id_dict，将原始数据中的 _raw_id 映射到从 _start_id 开始的整数 ID

        Args:
            _id_dict: 映射字典，可能会被填充内容
            _raw_id: 待映射的 ID
            _start_id: ID 从多少开始计算

        Returns:
            映射后的 ID
        """
        if _raw_id not in _id_dict:
            _id_dict[_raw_id] = len(_id_dict) + _start_id  # 填充新的 id
        return _id_dict[_raw_id]

    if project_path.run_env == 'local':
        write_mode = 'w'
    else:
        write_mode = 't'

    # 原始数据中的 id 映射到从 1 开始的 ID
    user_id_dict = dict()
    item_id_dict = dict()

    with gfile.GFile(input_fp) as input_f:
        with gfile.GFile(output_fp, write_mode) as output_f:
            output_f = csv.writer(output_f)
            output_f.writerow(['user_id', 'item_id', 'rating', 'timestamp'])
            for input_line_idx, input_line in enumerate(input_f):
                record = json.loads(input_line)

                user_id = _get_id(user_id_dict, record['reviewerID'])
                item_id = _get_id(item_id_dict, record['asin'])
                rating = record['overall']
                timestamp = record['unixReviewTime']

                output_f.writerow([user_id, item_id, rating, timestamp])

                if input_line_idx % 10000 == 0:
                    print('{} lines written'.format(input_line_idx + 1))

    # noinspection PyTypeChecker
    json.dump({'user': user_id_dict, 'item': item_id_dict}, gfile.GFile(mapping_fp, 'w'))


def main():
    data_fd = osp.join(project_path.data_fd, 'Amazon', 'Electronics_5')
    raw_data_fp = osp.join(data_fd, 'raw.txt')

    # 提取有效信息
    rating_only_data_fp = osp.join(data_fd, 'rating_only.csv')
    mapping_fp = osp.join(data_fd, 'id_map.json')
    if not gfile.Exists(rating_only_data_fp):
        extract_data(raw_data_fp, rating_only_data_fp, mapping_fp)

    # 将同一用户的数据整合到一起
    uid_to_data = dict()
    with gfile.GFile(rating_only_data_fp) as input_f:
        input_f = csv.reader(input_f)
        input_f.__next__()  # 跳过表头
        for user_id, item_id, rating, timestamp in input_f:
            if user_id not in uid_to_data:
                uid_to_data[user_id] = []
            uid_to_data[user_id].append([user_id, item_id, rating, timestamp])
    print('data loaded as dict uid_to_data')

    # 分用户写数据文件
    # ${user_id},${item_id},${item_id_seq},${rating_seq},${rating},${timestamp}
    processed_data_fd = osp.join(data_fd, 'processed')
    gfile.MakeDirs(processed_data_fd)
    processed_count = 0
    while uid_to_data:
        user_id, records = uid_to_data.popitem()
        records.sort(key=lambda x: int(x[-1]))  # 根据时间戳排序
        rating_seq = []
        item_id_seq = []
        with gfile.GFile(osp.join(processed_data_fd, '{}.csv'.format(user_id)), 'w') as output_f:
            output_f = csv.writer(output_f)
            for user_id, item_id, rating, timestamp in records:
                output_f.writerow([user_id, item_id, ' '.join(item_id_seq), ' '.join(rating_seq), rating, timestamp])
                rating_seq.append(rating)
                item_id_seq.append(item_id)
        if len(uid_to_data) % 100 == 0:
            print('separating data, {} users processed, {} users remained'.format(processed_count, len(uid_to_data)))
        processed_count += 1


if __name__ == '__main__':
    main()
