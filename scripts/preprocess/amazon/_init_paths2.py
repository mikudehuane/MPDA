# -*- coding: utf-8 -*-
# @Time    : 2021/1/4 3:47 下午
# @Author  : islander
# @File    : _init_paths.py
# @Software: PyCharm

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..', '..')
add_path(lib_path)
