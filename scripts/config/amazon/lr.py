# -*- coding: utf-8 -*-
# @Time    : 2021/8/28 下午12:49
# @Author  : islander
# @File    : din.py
# @Software: PyCharm


# 目前配置与 din 可以共用
from copy import deepcopy

from .din import *

FEA_CONFIG = deepcopy(FEA_CONFIG)  # 值拷贝避免共用对象
