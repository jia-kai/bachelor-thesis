# -*- coding: utf-8 -*-
# $File: workerobj.py
# $Date: Sat Mar 28 11:06:19 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import functools
import multiprocessing
import atexit
import weakref

class _bound_worker_method(object):
    __slots__ = ['func', 'instance', 'owner']

    def __init__(self, func, instance, owner):
        self.func = func
        self.instance = instance
        self.owner = owner

    def __call__(self, *args, **kwargs):
        return self.func(self.instance, *args, **kwargs)

    def __str__(self):
        return '<bound worker method {}.{} of {}>'.format(
            self.owner.__name__, self.func.__name__, self.instance)

    def __repr__(self):
        return str(self)


class AsyncTask(object):
    queue = None
    tid = None

    __noresult = object()

    _result = __noresult

    def __init__(self, queue, tid):
        self.queue = queue
        self.tid = tid

    def wait(self):
        """block master thread and wait for result"""
        if self._result is not self.__noresult:
            return self._result

        tid, result = self.queue.get()
        if tid != self.tid:
            raise RuntimeError(
                'bad task id: expect={} got={}; maybe some results are not'
                'waited on?'.format(self.tid, tid))
        self._result = result
        return result


class worker_method(object):
    """define a class method to be a worker method"""

    func = None

    def __init__(self, func):
        self.func = func
        self.__module__ = func.__module__
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, instance, owner):
        if instance is None:
            return self.func
        return _bound_worker_method(self.func, instance, owner)


class WorkerObj(object):
    """base class for an object that is intended to be called from master and
    do the work in a separate worker process"""

    __PROC_TYPE_UNINIT = -1
    __PROC_TYPE_MASTER = 0
    __PROC_TYPE_WORKER = 1
    __PROC_TYPE_MASTER_WORKER_STOPPED = 2

    __proc_type = __PROC_TYPE_UNINIT
    __task_queue = None
    """task queue; elements: (tid, name, args, kwargs)"""
    __result_queue = None
    """result queue; elements: (tid, result)"""
    __task_id = 0

    __worker_handle = None
    __worker_init_func = None

    def __init__(self):
        self.__task_queue = multiprocessing.Queue()
        self.__result_queue = multiprocessing.Queue()

    def __del__(self):
        self.stop_worker()

    def __getattribute__(self, name):
        val = object.__getattribute__(self, name)
        if (isinstance(val, _bound_worker_method) and
                self.__proc_type == self.__PROC_TYPE_MASTER):
            def func(_, *args, **kwargs):
                tid = self.__task_id
                self.__task_queue.put((tid, name, args, kwargs))
                self.__task_id += 1
                return AsyncTask(self.__result_queue, tid)
            val.func = func
        return val

    def start_worker(self, worker_init=None):
        """called in master to start the worker process"""
        assert self.__proc_type == self.__PROC_TYPE_UNINIT
        self.__worker_init_func = worker_init
        subp = multiprocessing.Process(target=self.__worker_proc)
        self.__worker_handle = subp
        subp.start()
        self.__proc_type = self.__PROC_TYPE_MASTER
        atexit.register(self.__stop_worker_on_exit, weakref.ref(self))
        del self.__worker_init_func

    def assert_worker_alive(self):
        assert self.__proc_type == self.__PROC_TYPE_MASTER
        assert self.__worker_handle.is_alive()

    def stop_worker(self):
        if self.__proc_type != self.__PROC_TYPE_MASTER:
            assert self.__proc_type in (
                self.__PROC_TYPE_UNINIT,
                self.__PROC_TYPE_MASTER_WORKER_STOPPED)
            return False

        self.__worker_handle.terminate()
        self.__worker_handle.join()
        self.__proc_type = self.__PROC_TYPE_MASTER_WORKER_STOPPED
        return True

    def __worker_proc(self):
        if self.__worker_init_func is not None:
            self.__worker_init_func()
        del self.__worker_init_func
        self.__proc_type = self.__PROC_TYPE_WORKER
        while True:
            tid, name, args, kwargs = self.__task_queue.get()
            func = getattr(self, name)
            result = func(*args, **kwargs)
            self.__result_queue.put((tid, result))

    @staticmethod
    def __stop_worker_on_exit(ref):
        obj = ref()
        obj.stop_worker()


def test():
    import os, time

    class Plus(WorkerObj):
        @worker_method
        def plus_w(self, a, b):
            time.sleep(1)
            print('plus_w-{}:'.format(os.getpid()), a, b)
            return a + b

        def plus(self, a, b):
            print('plus-{}:'.format(os.getpid()), a, b)
            return a + b

    p = Plus()
    assert p.plus(2, 3) == 5
    assert p.plus_w(6, 8) == 14
    def worker_init():
        print('init worker in {}'.format(os.getpid()))
    p.start_worker(worker_init=worker_init)
    assert p.plus(9, 8) == 17
    task = p.plus_w(2, 4)
    print ('task issued', task)
    assert task.wait() == 6
    print('done')

if __name__ == '__main__':
    test()
