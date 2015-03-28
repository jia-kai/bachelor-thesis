# -*- coding: utf-8 -*-
# $File: master.py
# $Date: Sat Mar 28 15:44:32 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .common import ISAParam, SharedValue
from .worker import FastISAWorker
from ..op import floatX

import numpy as np
import scipy.linalg

import multiprocessing
import logging
logger = logging.getLogger(__name__)

class FastISA(object):
    _isa_param = None
    _shared_val = None
    _workers = None
    _nr_data = None

    def __init__(self, isa_param, data, nr_worker, gpu_list=None):
        """
        :param data: numpy matrix of shape (nr_data, in_dim)
        :param gpu_list: if not None, specify the gpus of each worker"""
        assert isinstance(isa_param, ISAParam)
        assert data.ndim == 2 and data.shape[1] == isa_param.in_dim

        self._isa_param = isa_param
        self._shared_val = SharedValue(isa_param)
        self._nr_data = data.shape[0]

        assert nr_worker > 0
        if gpu_list is None:
            gpu_list = [None] * nr_worker
        else:
            assert len(gpu_list) == nr_worker
        self._workers = []
        for i in range(nr_worker):
            start = i * data.shape[0] / nr_worker
            end = (i + 1) * data.shape[0] / nr_worker
            worker = FastISAWorker(
                self._isa_param, self._shared_val, data[start:end])
            worker.start_worker(gpu_list[i])
            self._workers.append(worker)

    def get_result(self):
        self._init_whiten()
        self._init_whiten()
        return self._shared_val.data_whitening

    def _init_whiten(self):
        # calc mean and sustract by mean
        logger.info('calc data mean')
        acc = self._shared_val.result_accum
        acc.reset()
        self._invoke_workers(lambda i: i.accum_data_sum())
        self._shared_val.data_mean[:] = acc.get() / self._nr_data
        self._invoke_workers(lambda i: i.sub_by_mean())

        # calc cov matrix
        logger.info('calc data cov')
        acc.reset()
        self._invoke_workers(lambda i: i.accum_data_cov())
        cov = acc.get().reshape(self._isa_param.in_dim, self._isa_param.in_dim)
        cov /= self._nr_data
        self._shared_val.data_whitening[:] = cov

        # get whitening matrix
        logger.info('calc whitening matrix')
        self._shared_val.data_whitening_inv[:] = scipy.linalg.sqrtm(cov)
        self._shared_val.data_whitening[:] = scipy.linalg.inv(
            self._shared_val.data_whitening_inv)

        self._invoke_workers(lambda i: i.mul_by_whiten())


    def _invoke_workers(self, func):
        hdl = []
        for i in self._workers:
            hdl.append(func(i))
        for i in hdl:
            i.wait()
