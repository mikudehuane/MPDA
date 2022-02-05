# -*- coding: utf-8 -*-
# @Time    : 2021/7/22 上午11:48
# @Author  : islander
# @File    : logging_config.py
# @Software: PyCharm

import logging
import sys


class LoggingWhitelist(logging.Filter):
    def __init__(self, *whitelist):
        super().__init__()
        self._whitelist = [logging.Filter(name) for name in whitelist]

    def filter(self, record):
        return any(f.filter(record) for f in self._whitelist)


class LoggingBlacklist(LoggingWhitelist):
    def filter(self, record):
        return not LoggingWhitelist.filter(self, record)


logging.basicConfig(stream=sys.stdout, datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s [%(levelname)s] (%(name)s) - %(message)s',
                    level=logging.DEBUG)
for handler in logging.root.handlers:
    handler.addFilter(LoggingBlacklist('odps', 'urllib3', 'chardet'))
