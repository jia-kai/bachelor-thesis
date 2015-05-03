# -*- coding: utf-8 -*-
# $File: master.py
# $Date: Sun May 03 16:49:52 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .common import ISAParam, SharedValue
from .worker import ISAWorker
from .model import ISAModel
from ..op import floatX

import numpy as np

from collections import OrderedDict
import multiprocessing
import logging
import time
logger = logging.getLogger(__name__)

class ISA(object):
    _isa_param = None
    _shared_val = None
    _workers = None
    _nr_data = None

    def __init__(self, isa_param, data, nr_worker, gpu_list=None):
        """
        :param data: numpy matrix of shape (in_dim, nr_data)
        :param gpu_list: if not None, specify the gpus of each worker"""
        assert isinstance(isa_param, ISAParam)
        assert data.ndim == 2 and data.shape[0] == isa_param.in_dim

        self._isa_param = isa_param
        self._shared_val = SharedValue(isa_param)
        self._nr_data = data.shape[1]

        assert nr_worker > 0
        if gpu_list is None:
            gpu_list = [None] * nr_worker
        else:
            assert len(gpu_list) == nr_worker
        self._workers = []
        for i in range(nr_worker):
            start = i * data.shape[1] / nr_worker
            end = (i + 1) * data.shape[1] / nr_worker
            worker = ISAWorker(
                self._isa_param, self._shared_val,
                data[:, start:end], data.shape[1])
            worker.start_worker(gpu_list[i])
            self._workers.append(worker)

        self._init_whiten()
        self._init_weight()
        logger.info('ISA solver started, nr_worker={}'.format(nr_worker))

    def perform_iter(self, learning_rate):
        """perform one iteration
        :return: monitor channels as OrderedDict"""
        tstart = time.time()

        acc = self._shared_val.result_accum
        w = self._shared_val.isa_weight

        monitor = OrderedDict()

        monitor['learning_rate'] = learning_rate

        # compute grad
        acc.reset()
        self._invoke_workers(lambda i: i.accum_grad())
        delta = acc.get()
        delta *= learning_rate
        monitor['RMS[delta]'] = float(np.sqrt(np.square(delta).mean()))

        # update weight
        w -= delta.reshape(w.shape)
        self._normalize_weight()

        # calc cost
        acc.reset()
        self._invoke_workers(lambda i: i.accum_cost())
        cost = acc.get()
        assert cost.size == 1
        monitor['cost'] = float(cost[0])
        monitor['training_time'] = time.time() - tstart
        return monitor

    def _init_whiten(self):
        acc = self._shared_val.result_accum

        # calc mean and sustract by mean
        logger.info('calc data mean')
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

        # eigen decomposition
        eig_val, eig_vec = np.linalg.eigh(cov)
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]

        w = np.zeros_like(self._shared_val.data_whitening)
        logger.info('eigen values: {}'.format(eig_val))
        thresh = eig_val.sum() * self._isa_param.pca_energy_keep
        idx = 0
        while idx < self._isa_param.hid_dim:
            cur_ev = eig_val[idx]
            assert cur_ev >= self._isa_param.min_eigen, (cur_ev, idx)
            w[idx] = eig_vec[:, idx] / np.sqrt(cur_ev)
            thresh_next = thresh - cur_ev
            if thresh > 0 and thresh_next <= 0:
                logger.warn(
                    'energy too small at the {}th dimension; '
                    'thresh={} ev_neighbour={}'.format(
                        idx, thresh, eig_val[idx:idx+2]))
            thresh = thresh_next
            idx += 1
        logger.info('smallest eigen value: {}'.format(cur_ev))
        if idx < w.shape[0]:
            w = w[:idx]
            self._shared_val.reset_in_dim(idx)
            self._invoke_workers(lambda i: i.reset_in_dim(idx))
        logger.info('reduced input dimension: {}; hid dim: {}'.format(
            w.shape[0], self._isa_param.hid_dim))
        self._shared_val.data_whitening[:] = w
        self._invoke_workers(lambda i: i.mul_by_whiten())

    def _invoke_workers(self, func):
        hdl = []
        for i in self._workers:
            hdl.append(func(i))
        for i in hdl:
            i.wait()

    def _init_weight(self):
        rng = np.random.RandomState(19931102)
        w = self._shared_val.isa_weight
        w[:] = rng.uniform(size=w.shape)
        self._normalize_weight()

    def _normalize_weight(self):
        w = self._shared_val.isa_weight
        wwt = w.dot(w.T)
        eig_val, eig_vec = np.linalg.eigh(wwt)
        w[:] = eig_vec.dot(np.diag(1.0 / np.sqrt(eig_val))).dot(
            eig_vec.T).dot(w)

    def _do_get_model(self, coeff):
        coeff = np.array(coeff)
        return ISAModel(
            coeff=coeff,
            bias=-coeff.dot(self._shared_val.data_mean),
            outhid_conn=self._isa_param.make_outid_conn_mat())

    def get_model(self):
        sv = self._shared_val
        coeff = np.array(sv.isa_weight.dot(sv.data_whitening))
        return self._do_get_model(coeff)

    def get_model_pcaonly(self):
        return self._do_get_model(self._shared_val.data_whitening)
