# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Sat Mar 28 22:46:31 2015 +0800
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

class ProgressReporter(object):
    """report progress of long-term jobs"""
    _start_time = None
    _prev_report_time = 0
    _cnt = 0
    _name = None
    _total = None

    def __init__(self, name, total=0, fout=sys.stderr):
        self._start_time = time.time()
        self._name = name
        self._total = int(total)
        self._fout = fout

    @property
    def total_time(self):
        return time.time() - self._start_time

    def trigger(self, delta=1, extra_msg='', target_cnt=None):
        if target_cnt is None:
            self._cnt += int(delta)
        else:
            self._cnt = int(target_cnt)
        now = time.time()
        if now - self._prev_report_time < 0.5:
            return
        self._prev_report_time = now
        dt = now - self._start_time
        if self._total and self._cnt > 0:
            eta_msg = '{}/{} ETA: {:.2f}'.format(self._cnt, self._total,
                    (self._total-self._cnt)*dt/self._cnt)
        else:
            eta_msg = '{} done'.format(self._cnt)
        self._fout.write(
            '\r{}: avg {:.2f}/sec, passed {:.2f}sec, {}  {} '.format(
                self._name, self._cnt / dt, dt, eta_msg, extra_msg))
        self._fout.flush()

    def finish(self):
        """:return: total time"""
        self._fout.write('\n')
        return self.total_time

    def newline(self):
        """move to new line so others could write to output file"""
        self._fout.write('\n')
