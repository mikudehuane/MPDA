# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 下午2:36
# @Author  : islander
# @File    : utils.py
# @Software: PyCharm

import hashlib
import math


def get_array_slice(array, slice_count, *, should_order=True, order_seed=0):
    """将数组尽可能等分成 slice_count 个切片

    Args:
        array: 待切分数组
        slice_count: 目标分片数量
        should_order: 切分前是否一致化数组，因为有时多 worker 并行，调用 gfile.ListDirectory 弄出来的数组会不一致
            一致化的过程是，先排序，再用确定的种子 shuffle，从而保证一致并且各个分片不会有倾向
        order_seed: 一致化数组时，shuffle 的种子

    Returns:
        分片后数组的数组，其中可能存在空数组
    """
    import random

    if slice_count < 1:
        raise ValueError('slice_count should be at lease 1, but got {}'.format(slice_count))

    if should_order:
        array.sort()
        randomer = random.Random(order_seed)
        randomer.shuffle(array)

    num_entries_per_slice = len(array) / float(slice_count)
    num_entries_per_slice = int(math.ceil(num_entries_per_slice) + 1e-7)  # ensure to get the ceil of ratio

    ret = []
    for slice_idx in range(slice_count):  # 切出前面的分片
        start = min(slice_idx * num_entries_per_slice, len(array))
        end = min((slice_idx + 1) * num_entries_per_slice, len(array))
        slice_arr = array[start: end]
        ret.append(slice_arr)
    return ret


def count_parameters(tensor_list):
    parameter_count = 0
    for tensor in tensor_list:
        _parameter_count = 1
        for dim in tensor.get_shape().as_list():
            _parameter_count *= dim
        parameter_count += _parameter_count
    return parameter_count


def parse_fp(inp):  # 允许出现 project_fd, data_fd, log_fd 作为指代
    import project_path
    inp = inp.replace('{project_fd}', project_path.project_fd)
    inp = inp.replace('{data_fd}', project_path.data_fd)
    inp = inp.replace('{log_fd}', project_path.log_fd)
    inp = inp.replace('{output_fd}', project_path.output_fd)
    return inp


def get_user_ids(data_fds, require='list'):
    """获取所有的 user_id

    Args:
        require: list/set，指定返回的类型，默认返回列表
        data_fds (List[str]): 数据存放的目录的列表，函数会取这些目录下文件名的并集来提取 user_id 列表

    Returns:
        List[str]: 所有 user_id 的列表，顺序不保证
    """
    from tensorflow.python.platform import gfile
    import re

    pat = re.compile(r'(\d+)\.csv')  # 用户数据文件名的格式
    user_ids = set()  # 存放结果
    for data_fd in data_fds:
        fns = gfile.ListDirectory(data_fd)
        for fn in fns:
            match_obj = re.match(pat, fn)
            if match_obj:
                user_ids.add(match_obj.group(1))
    if require == 'list':
        user_ids = list(user_ids)
    else:
        assert require == 'set'
    return user_ids


def get_ordered_dict(input_dict, key_order):  # 返回按照 order 排序的 OrderedDict
    from collections import OrderedDict
    ret = OrderedDict()
    for key in key_order:
        ret[key] = input_dict[key]
    return ret


def certain_hash(string):  # 返回一个字符串的哈希值，且每次运行返回相同值
    return int(hashlib.md5(string.encode('utf-8')).hexdigest()[:15], 16)


def parse_args_warn_on_verbose(parser, **kwargs):
    """解析命令行参数，当有不匹配时打印警告

    Args:
        parser: 解析器
        **kwargs: parse_known_args 的参数

    Returns:

    """
    args, unparsed_args = parser.parse_known_args(**kwargs)

    if unparsed_args:
        print('WARNING: Found unrecognized sys.argv: {}'.format(unparsed_args))

    return args


def get_id_and_update_dict(id_dict, raw_id, start_id=1):
    """根据字典 id_dict，将原始数据中的 raw_id 映射到从 start_id 开始的整数 ID

    Args:
        id_dict: 映射字典，可能会被填充内容
        raw_id: 待映射的 ID
        start_id: ID 从多少开始计算

    Returns:
        映射后的 ID
    """
    if raw_id not in id_dict:
        id_dict[raw_id] = len(id_dict) + start_id  # 填充新的 id
    return id_dict[raw_id]


def robust_geq(larg, rarg, eps=1e-7):
    """算数鲁邦的 >=，当 larg > rarg-eps 时，即返回 True
    """
    return larg > rarg - eps
