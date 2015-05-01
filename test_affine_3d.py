#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: test_affine_3d.py
# $Date: Fri May 01 13:23:58 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.math.op.affine3d import batched_affine3d, make_rot3d_uniform
from nasmia.utils import serial
from nasmia.visualize import view_3d_data_simple

import argparse

import numpy as np
import time

def main():
    parser = argparse.ArgumentParser(
        description='test batched 3d affine transform',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input')
    parser.add_argument('-b', '--batch', type=int, default=0,
                        help='batch number to view')
    args = parser.parse_args()

    inp = serial.load(args.input)
    assert inp.ndim == 4

    #affine_mat = np.tile(np.eye(4), (inp.shape[0], 1, 1))

    zl, yl, xl = inp.shape[1:]
    affine_mat = make_rot3d_uniform(inp.shape[0],
                                    center=(xl / 2, yl / 2, zl / 2))
    out = batched_affine3d(inp, affine_mat)
    view_3d_data_simple(out[args.batch])

    return # do not benchmark
    nr_time = 10
    t0 = time.time()
    for i in range(nr_time):
        out1 = batched_affine3d(inp, affine_mat)
        assert np.max(np.abs(out - out1)) < 1e-9
    dt = (time.time() - t0) / nr_time
    print 'time={:.3f}ms GFlops={:.3f}'.format(
        dt * 1000,
        out.size * (28 + 21) / dt / 1e9)

if __name__ == '__main__':
    main()
