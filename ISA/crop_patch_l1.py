#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: crop_patch_l1.py
# $Date: Sun May 10 20:56:51 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial, ProgressReporter
from nasmia.utils.patch_cropper import CropPatchHelper
from nasmia.math.ISA.config import LAYER1_PATCH_SIZE

import numpy as np
import theano.tensor as T
import theano

import argparse
import os
import logging
logger = logging.getLogger(__name__)

class PatchCropper(object):
    _prog = None
    _helper = None
    _output = None
    _output_idx = 0

    _fprop_inp = None
    _fprop_l0 = None

    def __init__(self, args):
        basedir = os.path.dirname(args.imglist)
        with open(args.imglist) as fin:
            flist = fin.readlines()

        out_patch_tot_size = self._init_l0(args)

        self._output = np.empty(
            (out_patch_tot_size, len(flist) * args.patch_per_img),
            dtype='float32')
        logger.info('data size: {:.3f} GiB'.format(
            self._output.size * 4 / 1024.0**3))

        self._prog = ProgressReporter('crop', self.nr_patch)
        self._helper = CropPatchHelper(
            args.patch_size, 0, rng=np.random.RandomState(args.seed))

        for fpath in flist:
            fpath = fpath.strip()
            data = serial.load(os.path.join(basedir, fpath))
            self._work_single(data, args)
        self._prog.finish()

        serial.dump(self._output, args.output)

    def _init_l0(self, args):
        model = serial.load(args.l0_model)
        x = T.TensorType('floatX', (False, ) * 5)('x')
        kern_shape = model.conv_kern_shape
        ks = kern_shape[0]
        assert all(i == ks for i in kern_shape)
        y = model.fprop_conv(x, stride=args.patch_size - ks)
        f = theano.function([x], y)
        out_patch_tot_size = model.out_chl * 8
        def fprop():
            y = f(self._fprop_inp)
            assert y.ndim == 5 and y.shape[2:] == (2, 2, 2)
            return y.reshape(args.patch_per_img, out_patch_tot_size)
        self._fprop_l0 = fprop
        ps = args.patch_size
        self._fprop_inp = np.empty(
            shape=(args.patch_per_img, 1, ps, ps, ps), dtype='float32')
        return out_patch_tot_size

    @property
    def nr_patch(self):
        return self._output.shape[1]

    def _work_single(self, data, args):
        it = self._helper(data)

        for i in range(args.patch_per_img):
            self._fprop_inp[i, 0] = next(it)

        s = self._output_idx
        t = s + args.patch_per_img
        self._output[:, s:t] = self._fprop_l0().T
        self._output_idx = t
        self._prog.trigger(t - s)


def main():
    parser = argparse.ArgumentParser(
        description='randomly crop patches from nrrd image files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--patch_size', type=int, default=LAYER1_PATCH_SIZE)
    parser.add_argument('--patch_per_img', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=20150405,
                        help='rng seed')
    parser.add_argument('l0_model')
    parser.add_argument('imglist', help='path to list of images')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args()

    PatchCropper(args)

if __name__ == '__main__':
    main()
