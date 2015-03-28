#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: crop_patch.py
# $Date: Sat Mar 28 23:28:09 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.thirdparty import nrrd
from nasmia.utils import serial, ProgressReporter
from nasmia.visualize import view_3d_data_simple

import numpy as np

import argparse
import os
import logging
logger = logging.getLogger(__name__)

class PatchCropper(object):
    _prog = None
    _rng = None
    _output = None
    _output_idx = 0

    def __init__(self, args):
        basedir = os.path.dirname(args.imglist)
        with open(args.imglist) as fin:
            flist = fin.readlines()

        self._output = np.empty(
            (args.patch_size**3, len(flist) * args.patch_per_img),
            dtype='float32')
        logger.info('data size: {:.3f} GiB'.format(
            self._output.size * 4 / 1024.0**3))

        self._prog = ProgressReporter('crop', self.nr_patch)
        self._rng = np.random.RandomState(args.seed)

        for fpath in flist:
            fpath = fpath.strip()
            data, header = nrrd.read(os.path.join(basedir, fpath))
            self._work_single(data, args)
        self._prog.finish()

        serial.dump(self._output, args.output)

    @property
    def nr_patch(self):
        return self._output.shape[1]

    def _work_single(self, data, args):
        thresh = data.mean() * 0.8
        axrange = self._find_axis_range(data, thresh)
        for idx, (i, j) in enumerate(axrange):
            assert j - i > args.patch_size * 2
            axrange[idx] = (i, j - args.patch_size + 1)

        def gen_subpatch():
            r = lambda v: self._rng.randint(*v)
            sub = [slice(v, v + args.patch_size)
                   for v in map(r, axrange)]
            return data[tuple(sub)]

        for i in range(args.patch_per_img):
            while True:
                sub = gen_subpatch()
                if sub.mean() >= thresh:
                    break
            self._output[:, self._output_idx] = sub.flat
            self._output_idx += 1
            self._prog.trigger()

    def _find_axis_range(self, data, thresh):
        rst = []
        for axis in range(data.ndim):
            def visit(v):
                idx = [slice(None)] * data.ndim
                idx[axis] = v
                tup = tuple(idx)
                return data[tup]
            low = 0
            while visit(low).mean() < thresh:
                low += 1
            high = data.shape[axis] - 1
            while visit(high).mean() < thresh:
                high -= 1
            assert low < high
            rst.append((low, high + 1))
        return rst


def main():
    parser = argparse.ArgumentParser(
        description='randomly crop patches from nrrd image files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--patch_size', type=int, default=13)
    parser.add_argument('--patch_per_img', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=20150328,
                        help='rng seed')
    parser.add_argument('imglist', help='path to list of images')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args()

    PatchCropper(args)

if __name__ == '__main__':
    main()
