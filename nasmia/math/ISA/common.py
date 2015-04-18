# -*- coding: utf-8 -*-
# $File: common.py
# $Date: Sun Apr 05 20:52:35 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from ..op import floatX
from .. import shmarray

import numpy as np

import multiprocessing

class ISAParam(object):
    in_dim = None
    out_dim = None

    hid_dim = None
    """dimensionality of hidden layer"""

    subspace_size = None
    """number of hidden layer nodes contained in one subspace"""

    pca_energy_keep = 0.7
    """total energy to keep for PCA dimension reduction and whitening"""

    min_eigen = 1e-8
    """minimum eigen value ensured during PCA"""

    eps = 1e-6

    def __init__(self, in_dim, subspace_size, hid_dim=None, out_dim=None):
        self.in_dim = int(in_dim)
        self.subspace_size = int(subspace_size)
        if not hid_dim and not out_dim:
            raise ValueError(
                'must specifiy at least one of hid_dim or out_dim')
        if hid_dim:
            hid_dim = int(hid_dim)
        if out_dim:
            out_dim = int(out_dim)

        if not hid_dim:
            hid_dim = out_dim * self.subspace_size
        if not out_dim:
            out_dim = hid_dim / self.subspace_size

        assert hid_dim == out_dim * self.subspace_size

        self.hid_dim = hid_dim
        self.out_dim = out_dim

    def make_outid_conn_mat(self):
        """:return: a matrix of shape(out_dim, hid_dim) to describe 
            the connection from hidden layer to output layer
        """
        rst = np.zeros((self.out_dim, self.hid_dim), dtype=floatX())
        s = 0
        val = 1.0 / self.subspace_size
        for i in range(self.out_dim):
            s1 = s + self.subspace_size
            rst[i, s:s1] = val
            s = s1
        return rst


class SharedAccumulator(object):
    _dim = None
    _arr = None
    _arr_size_store = None
    _lock = None

    def __init__(self, dim):
        dim = int(dim)
        self._dim = dim
        self._arr = shmarray.create((dim, ), dtype=floatX())
        self._arr_size_store = multiprocessing.RawValue('I')
        self._lock = multiprocessing.Lock()

    @property
    def _arr_size(self):
        return self._arr_size_store.value

    @_arr_size.setter
    def _arr_size(self, v):
        self._arr_size_store.value = v

    def reset(self):
        """reset accumulator to zero, called in master process"""
        self._arr[:] = 0
        self._arr_size = 0

    def get(self, lock_and_copy=False):
        """get current value"""
        assert self._arr_size
        if lock_and_copy:
            with self._lock:
                return self._arr[:self._arr_size].copy()
        return self._arr[:self._arr_size]

    def add(self, val):
        """add to the accumulator"""
        assert val.ndim == 1 and val.size <= self._dim
        with self._lock:
            if self._arr_size == 0:
                self._arr_size = val.size
            else:
                assert self._arr_size == val.size
            self._arr[:val.size] += val


class SharedValue(object):
    """values shared across processes on CPU memory"""

    data_mean = None
    """mean value of each channel of the data"""

    data_whitening = None
    """data whitening matrix"""

    isa_weight = None
    """(hid_dim, in_dim): left matrix to transform data into hidden layer"""

    result_accum = None
    """result accumulator, of shape in_dim * in_dim (maximally possible dim)"""

    __init_finished = False

    def __init__(self, isa_param):
        assert isinstance(isa_param, ISAParam)
        dtype = floatX()
        mk = lambda *shape: shmarray.create(shape=shape, dtype=dtype)
        self.data_mean = mk(isa_param.in_dim)
        self.data_whitening = mk(isa_param.in_dim, isa_param.in_dim)
        self.isa_weight = mk(isa_param.hid_dim, isa_param.in_dim)
        self.result_accum = SharedAccumulator(isa_param.in_dim ** 2)
        self.__init_finished = True

    def reset_in_dim(self, dim):
        """reset input dimension (used after dimension reduction)"""
        self.__init_finished = False
        def sub(arr, shape):
            assert np.prod(shape) < arr.size
            return shmarray.create(
                shape, dtype=arr.dtype, force_storage=arr.ctypesArray)

        self.data_whitening = sub(
            self.data_whitening,
            (dim, self.data_whitening.shape[1]))
        self.isa_weight = sub(
            self.isa_weight,
            (self.isa_weight.shape[0], dim))

        self.__init_finished = True

    def __setattr__(self, name, val):
        if not self.__init_finished or name == '_SharedValue__init_finished':
            return super(SharedValue, self).__setattr__(name, val)
        raise ValueError('could not set attribute {}'.format(name))
