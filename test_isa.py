#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: test_isa.py
# $Date: Sat Mar 28 21:38:03 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.math.ISA import ISA, ISAParam

import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

def gen_data(param, nr_data):
    rng = np.random.RandomState(19930501)
    hid_data = rng.normal(size=(param.hid_dim, nr_data))

    s = 0
    for i in range(param.out_dim):
        s1 = s + param.subspace_size
        hid_data[s:s1] *= rng.uniform(size=(1, nr_data))
        s = s1
    assert s == param.hid_dim
    mixing = rng.uniform(size=(param.in_dim, param.hid_dim))
    #visualize(np.corrcoef(np.square(hid_data)))
    return np.dot(mixing, hid_data)

def visualize(mat, repermute=False):
    if repermute:
        for i in range(mat.shape[0]):
            cols = mat[i, i:]
            _, idx = zip(*sorted([(-v, c + i) for c, v in enumerate(cols)]))
            idx = range(i) + list(idx)
            mat = mat[idx]
            mat = mat[:, idx]
    plt.matshow(mat, cmap='gray')
    plt.show()

def main():
    nr_data = 10000
    param = ISAParam(in_dim=60, subspace_size=4, hid_dim=40)
    data = gen_data(param, nr_data)
    isa = ISA(param, data, nr_worker=2)
    dcheck = isa._shared_val.data_whitening.dot(
        data - isa._shared_val.data_mean.reshape(-1, 1))
    assert np.abs(np.cov(dcheck) - np.eye(dcheck.shape[0])).max() <= 1e-2
    for i in range(100):
        monitor = isa.perform_iter(30)
        msg = 'train iter {}\n'.format(i)
        for k, v in monitor.iteritems():
            msg += '{}: {}\n'.format(k, v)
        logger.info(msg[:-1])

    visualize(np.corrcoef(isa.apply_to_data(data, do_reduce=False)))

if __name__ == '__main__':
    main()
