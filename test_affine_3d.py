#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: test_affine_3d.py
# $Date: Tue Jun 09 15:50:03 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.math.op.affine3d import batched_affine3d, RandomAffineMat
from nasmia.utils import serial
from nasmia.visualize import view_3d_data_single

import numpy as np

import argparse
import time
import math

def main():
    parser = argparse.ArgumentParser(
        description='test batched 3d affine transform',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--load',
                        help='load dumped argument')
    parser.add_argument('-i', '--input',
                        help='batch image input')
    parser.add_argument('--gen', action='store_true',
                        help='generate data for test')
    parser.add_argument('--benchmark', action='store_true',
                        help='test kernel speed')
    parser.add_argument('-b', '--batch', type=int, default=0,
                        help='batch number to view')
    parser.add_argument('--border', type=int, default=40,
                        help='border to remove')
    parser.add_argument('--angle', type=float, default=30,
                        help='max rotate angle in degrees')
    parser.add_argument('-o', '--output', help='write output to file')
    parser.add_argument('--identity', action='store_true',
                        help='use identity transform')
    args = parser.parse_args()

    add_axis = False
    if args.gen:
        inp = np.array([np.random.uniform(size=(4, 2, 3))] * 3)
        affine_mat = np.eye(4)[:3]
        oshp = inp.shape[1:]
    elif args.input:
        inp = serial.load(args.input)
        if inp.ndim == 3:
            add_axis = True
            inp = np.expand_dims(inp, axis=0)
        assert inp.ndim == 4

        if args.identity:
            affine_mat = np.tile(np.eye(4), (inp.shape[0], 1, 1))[:, :3]
            oshp = inp.shape[1:]
        else:
            xl, yl, zl = inp.shape[1:]
            affine_mat = RandomAffineMat(
                inp.shape[0], center=(xl / 2, yl / 2, zl / 2),
                min_angle=math.radians(-args.angle),
                max_angle=math.radians(args.angle))()
            affine_mat[:, :, 3] += args.border / 2
            oshp = (xl - args.border, yl - args.border, zl - args.border)
            if affine_mat.shape[0] == 1:
                print 'affine mat:', affine_mat[0]
    else:
        assert args.load, 'must provide -i or -l'
        inp, affine_mat, oshp, orig_out = serial.load(args.load)
    out = batched_affine3d(inp, affine_mat, oshp)

    if args.output:
        out1 = out
        if add_axis:
            out1 = np.squeeze(out, axis=0)
        serial.dump(out1, args.output, use_pickle=True)

    if args.gen:
        print out
    elif not args.benchmark and not args.output:
        view_3d_data_single(out[args.batch])

    if args.benchmark:
        inp = np.ascontiguousarray(inp).astype(np.float32)
        affine_mat = np.ascontiguousarray(affine_mat).astype(np.float32)
        nr_time = 10
        t0 = time.time()
        for i in range(nr_time):
            out1 = batched_affine3d(inp, affine_mat, oshp)
            assert np.max(np.abs(out - out1)) < 1e-9
        dt = (time.time() - t0) / nr_time
        print 'time={:.3f}ms GFlops={:.3f}'.format(
            dt * 1000,
            out.size * (28 + 21) / dt / 1e9)

if __name__ == '__main__':
    main()
