# -*- coding: utf-8 -*-
# @Time    : 2021/7/30 上午11:01
# @Author  : islander
# @File    : cached_actions.py
# @Software: PyCharm


class CachedActions(object):
    """缓存一些操作（作为函数缓存），用于实现类似原子操作的回滚
    """

    def __init__(self):
        self.actions = []

    def add_action(self, func, args=None, kwargs=None):  # 添加一个操作，以函数和参数形式添加
        if args is None:
            args = []
        if kwargs is None:
            kwargs = dict()
        self.actions.append([func, args, kwargs])

    def apply_actions(self):
        for action in self.actions:
            action[0](*action[1], **action[2])

    def clear_actions(self):
        self.actions.clear()

    @property
    def num_actions(self):
        return len(self.actions)
