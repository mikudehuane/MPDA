# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 下午2:57
# @Author  : islander
# @File    : custom_logger.py
# @Software: PyCharm

import collections


class CustomLogger(object):
    """自定义的日志记录对象，封装一些日志记录操作

    - 记录日志时同时向控制台和文件输出

    Args:
        logger (logging.Logger): 用于打印控制台输出，使用 info 打印
    """

    def __init__(self, logger):
        self._logger = logger

    def log_text(self, msg, *, file_handler=None):
        """打印一条文本信息，并写入文件（写后空行）

        Args:
            file_handler: 文件读写对象，若不指定，则不写文件
            msg: 待记录的信息
        """
        self._logger.info(msg)
        if file_handler is not None:
            file_handler.write(msg)
            # 空一行
            if not msg.endswith('\n'):
                file_handler.write('\n\n')
            else:
                file_handler.write('\n')
            file_handler.flush()

    def log_dict(self, msg, hint=None, *, file_handler=None):
        """打印一个有序字典，日志以 key0: value0, key1: value1 形式输出，文件按 csv 格式写

        Args:
            hint: 打印日志时先打印该 hint（作为单独一行）
            msg (collections.OrderedDict): 待记录的字典
            file_handler: 文件读写对象，若不指定，不写文件
        """
        if hint is not None:
            self._logger.info(hint)
        print_msg = ', '.join(['{}: {}'.format(key, val) for key, val in msg.items()])
        self._logger.info(print_msg)

        if file_handler is not None:
            write_msg = ','.join([str(val) for val in msg.values()]) + '\n'
            file_handler.write(write_msg)
            file_handler.flush()
