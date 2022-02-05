# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 下午4:20
# @Author  : islander
# @File    : utils.py
# @Software: PyCharm

import sys


def get_command():
    command = 'python ' + ' '.join(sys.argv)
    return command
