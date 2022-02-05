# -*- coding: utf-8 -*-
# @Time    : 2021/7/14 下午5:56
# @Author  : islander
# @File    : __init__.py
# @Software: PyCharm

"""存放模型代码
"""

from . import din
from . import linear
from . import deepfm
from .din import train_by_net, evaluate_by_net  # 这个函数是模型无关的，并列于 din 导入
from .din import ShuffleReader
from . import wide_deep
from . import pnn
