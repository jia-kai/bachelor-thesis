# -*- coding: utf-8 -*-
# $File: worker.py
# $Date: Sat Mar 28 15:39:34 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .common import ISAParam, SharedValue
from ..op import floatX, sharedX, set_gpu_num
from ...utils import cached_property, timed_operation
from ...utils.workerobj import WorkerObj, worker_method

import numpy as np

import multiprocessing

# theano modules to be initialized in worker process
T = None
theano = None

class FastISAWorker(WorkerObj):
    _isa_param = None
    _shared_val = None

    _theano_shared_data = None
    """data in theano memory, where each sample is a column vector
    (shape: (dim, nr_data))"""

    _data = None
    nr_data = None

    def __init__(self, isa_param, shared_val, data):
        """:param data: original data, in row-vector format
            (shape: (nr_data, dim))"""
        super(FastISAWorker, self).__init__()
        assert isinstance(isa_param, ISAParam)
        assert isinstance(shared_val, SharedValue)
        self._isa_param = isa_param
        self._shared_val = shared_val
        self._data = data
        self.nr_data = data.shape[0]

    def start_worker(self, gpu_num=None):
        def worker_init():
            global T, theano
            if gpu_num is not None:
                set_gpu_num(gpu_num)
            import theano.tensor as T_
            import theano as theano_
            T = T_
            theano = theano_
            self._theano_shared_data = sharedX(self._data.T)
            del self._data

        super(FastISAWorker, self).start_worker(worker_init=worker_init)
        del self._data

    @worker_method
    def accum_data_sum(self):
        """accumulate the sum value"""
        self._shared_val.result_accum.add(self._theano_data_sum())

    @worker_method
    def accum_data_cov(self):
        """accumulate sum(x[i]'[j])"""
        self._shared_val.result_accum.add(self._theano_data_cov_sum())

    @worker_method
    def sub_by_mean(self):
        mean = sharedX(self._shared_val.data_mean).dimshuffle(0, 'x')
        v1 = self._theano_shared_data - mean
        f = theano.function(
            [], [], updates=[(self._theano_shared_data, v1)])
        f()

    @worker_method
    def mul_by_whiten(self):
        wht = sharedX(self._shared_val.data_whitening)
        v1 = T.dot(wht, self._theano_shared_data)
        f = theano.function(
            [], [], updates=[(self._theano_shared_data, v1)])
        f()

    @cached_property
    def _theano_data_sum(self):
        y = T.sum(self._theano_shared_data, axis=1)
        with timed_operation('compiling data_sum'):
            return theano.function([], y)

    @cached_property
    def _theano_data_cov_sum(self):
        y = T.dot(self._theano_shared_data, self._theano_shared_data.T)
        with timed_operation('compiling data_cov_sum'):
            return theano.function([], y.flatten())
