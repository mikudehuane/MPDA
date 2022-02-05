# -*- coding: utf-8 -*-
# @Time    : 2021/7/13 下午5:44
# @Author  : islander
# @File    : statistics_timestamp.py
# @Software: PyCharm

"""统计数据的时间戳分布
"""


def main():
    import _init_paths2
    import project_path
    import os.path as osp
    from tensorflow.python.platform import gfile
    import csv
    import argparse
    from gutils import parse_fp, parse_args_warn_on_verbose

    parser = argparse.ArgumentParser(description='统计数据的时间戳')
    parser.add_argument('-idp', '--input_data_fp', type=parse_fp,
                        default=osp.join(project_path.data_fd, 'ml-20m', 'ratings.csv'),
                        help='数据文件的路径')
    args = parse_args_warn_on_verbose(parser)

    timestamps = []
    with gfile.GFile(args.input_data_fp) as input_f:
        input_f = csv.reader(input_f)
        heading = input_f.__next__()
        heading = {name: idx for idx, name in enumerate(heading)}
        ts_col = heading['timestamp']

        for idx, line in enumerate(input_f):
            timestamp = int(line[ts_col])
            timestamps.append(timestamp)
            if idx % 10000 == 0:
                print('{} lines read'.format(idx))
        timestamps.sort()

        train_ratio = 0.75
        split_index = int(len(timestamps) * train_ratio)  # 分位点

        print('数据量：{}'.format(len(timestamps)))
        print('排序后以第 {} 条数据分界'.format(split_index))
        print('分界值为 {}'.format(timestamps[split_index]))


if __name__ == '__main__':
    main()
