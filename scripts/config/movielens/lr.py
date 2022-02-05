# -*- coding: utf-8 -*-
# @Time    : 2021/8/16 下午1:31
# @Author  : islander
# @File    : lr.py
# @Software: PyCharm

# 目前配置与 din 可以共用
from copy import deepcopy

from .din import *

FEA_CONFIG = deepcopy(FEA_CONFIG)  # 值拷贝避免共用对象

FEA_CONFIG['movie_genome'] = {
    'shape': (1128,),
    'category': 'val_vec',
}
