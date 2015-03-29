# -*- coding: utf-8 -*-
# $File: worker.py
# $Date: Sun Mar 29 09:42:31 2015 +0800
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

class ISAWorker(WorkerObj):
    _isa_param = None
    _shared_val = None

    _theano_shared_data = None
    """data in theano memory, where each sample is a column vector
    (shape: (dim, nr_data))"""

    _data = None
    _tot_nr_data = None

    def __init__(self, isa_param, shared_val, data, tot_nr_data):
        """:param data: original data, in column-vector format
            (shape: (dim, nr_data))"""
        super(ISAWorker, self).__init__()
        assert isinstance(isa_param, ISAParam)
        assert isinstance(shared_val, SharedValue)
        assert data.shape[0] == isa_param.in_dim
        self._isa_param = isa_param
        self._shared_val = shared_val
        self._data = data
        self._tot_nr_data = tot_nr_data

    def start_worker(self, gpu_num=None):
        def worker_init():
            global T, theano
            if gpu_num is not None:
                set_gpu_num(gpu_num)
            import theano.tensor as T_
            import theano as theano_
            T = T_
            theano = theano_
            self._theano_shared_data = sharedX(self._data)
            del self._data

        super(ISAWorker, self).start_worker(worker_init=worker_init)
        del self._data

    @worker_method
    def reset_in_dim(self, dim):
        self._shared_val.reset_in_dim(dim)

    @worker_method
    def accum_data_sum(self):
        """accumulate the sum value"""
        y = T.sum(self._theano_shared_data, axis=1)
        self._shared_val.result_accum.add(y.eval())

    @worker_method
    def accum_data_cov(self):
        """accumulate sum(x[i]'[j])"""
        y = T.dot(self._theano_shared_data,
                  self._theano_shared_data.T).flatten()
        self._shared_val.result_accum.add(y.eval())

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

    @worker_method
    def accum_grad(self):
        _, fgrad = self._theano_cost_grad
        grad = fgrad(self._shared_val.isa_weight)
        self._shared_val.result_accum.add(grad.flatten())

    @worker_method
    def accum_cost(self):
        fcost, _ = self._theano_cost_grad
        cost = fcost(self._shared_val.isa_weight).reshape((1, ))
        self._shared_val.result_accum.add(cost)

    @cached_property
    def _theano_cost_grad(self):
        """theano functions that takes W and return cost or grad_wrt_w"""
        conn_mat = sharedX(self._isa_param.make_outid_conn_mat())
        W = T.matrix()
        hidv = T.square(T.dot(W, self._theano_shared_data))
        reduced = T.dot(conn_mat, hidv)
        outv = T.sqrt(reduced + self._isa_param.eps)
        cost = outv.sum() * sharedX(
            1.0 / self._tot_nr_data / self._isa_param.out_dim)
        grad = T.grad(cost, W)
        with timed_operation('compiling grad and cost'):
            f_cost = theano.function([W], cost)
            f_grad = theano.function([W], grad)
            return f_cost, f_grad
