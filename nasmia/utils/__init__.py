# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Sat Mar 28 14:28:59 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from contextlib import contextmanager
import sys
import time

class cached_property(object):
    """property whose results is cached"""

    func = None

    def __init__(self, func):
        self.func = func
        self.__module__ = func.__module__
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__cache_key = '__result_cache_{}_{}'.format(
            func.__name__, id(func))

    def __get__(self, instance, owner):
        if instance is None:
            return self.func
        v = getattr(instance, self.__cache_key, None)
        if v is not None:
            return v
        v = self.func(instance)
        assert v is not None
        setattr(instance, self.__cache_key, v)
        return v

@contextmanager
def timed_operation(message):
    sys.stderr.write('start {} ...\n'.format(message))
    sys.stderr.flush()
    stime = time.time()
    yield
    sys.stderr.write('finished {}, time={:.2f}sec\n'.format(
        message, time.time() - stime))
