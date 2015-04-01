#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: test_isa.py
# $Date: Wed Apr 01 22:17:16 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.math.op import sharedX
from nasmia.math.ISA import ISA, ISAParam

import numpy as np
import matplotlib.pyplot as plt

import logging
import multiprocessing
import argparse
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

def eval_by_conv(data, model):
    data = data.T
    data = data.reshape([data.shape[0]] + list(model.get_conv_coeff().shape[1:]))
    y = model.fprop_conv(sharedX(data))
    return y.eval().reshape(data.shape[0], -1)

def main():
    parser = argparse.ArgumentParser(
        description='test ISA by synthetic data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--nr_data', type=int, default=10000)
    parser.add_argument('--gpus', help='comma separated list of gpus')
    parser.add_argument('--nr_iter', type=int, default=100)
    args = parser.parse_args()

    if args.gpus:
        gpus = map(int, args.gpus.split(','))
        isa_args = {'nr_worker': len(gpus), 'gpu_list': gpus}
    else:
        isa_args = {'nr_worker': multiprocessing.cpu_count()}

    param = ISAParam(in_dim=64, subspace_size=4, hid_dim=40)
    data = gen_data(param, args.nr_data)
    isa = ISA(param, data, **isa_args)
    dcheck = isa._shared_val.data_whitening.dot(
        data - isa._shared_val.data_mean.reshape(-1, 1))
    assert np.abs(np.cov(dcheck) - np.eye(dcheck.shape[0])).max() <= 1e-2
    for i in range(args.nr_iter):
        monitor = isa.perform_iter(30)
        msg = 'train iter {}\n'.format(i)
        for k, v in monitor.iteritems():
            msg += '{}: {}\n'.format(k, v)
        logger.info(msg[:-1])

    model = isa.get_model()
    cost_check = model(data).mean()
    assert abs(cost_check - monitor['cost']) < 1e-4
    if args.gpus:
        cost_check = eval_by_conv(data, model).mean()
        assert abs(cost_check - monitor['cost']) < 1e-4
    visualize(np.corrcoef(model(data, level2=False)))

if __name__ == '__main__':
    main()
