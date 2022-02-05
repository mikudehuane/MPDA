# -*- coding: utf-8 -*-
# @Time    : 2021/7/13 上午10:29
# @Author  : islander
# @File    : project_path.py
# @Software: PyCharm


import os.path as osp
import argparse

_parser = argparse.ArgumentParser('detect oss env', add_help=False)
_parser.add_argument('--buckets', default=None, type=str, help='OSS 用户根目录')
_args, _ = _parser.parse_known_args()

if _args.buckets is None:  # 本地环境
    run_env = 'local'
    project_fd = osp.normpath(osp.join(__file__, '..', '..'))  # 项目根目录
    data_fd = osp.join(r'/root/data/')  # 数据根目录
else:  # pai 环境
    run_env = 'pai'
    project_fd = osp.join(_args.buckets, 'Data-Sharing-Transfer')
    data_fd = osp.join(_args.buckets, 'data')  # OSS 上数据存储在根目录

output_fd = osp.join(project_fd, 'outputs')  # 输出文件的路径（如训练好的模型）
log_fd = osp.join(project_fd, 'log')  # 日志文件的存储路径
