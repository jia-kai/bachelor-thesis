#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: apply_model_on_data.py
# $Date: Sun Apr 19 21:49:29 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial, timed_operation, ProgressReporter
from nasmia.math.ISA.model import ISAModel
from nasmia.math.ISA.config import LAYER1_PATCH_SIZE, LAYER0_STRIDE

import theano.tensor as T
import theano
import numpy as np

import itertools
import logging
import argparse
import os
logger = logging.getLogger(__name__)


def make_fprop(layer0, layer1):
    assert isinstance(layer0, ISAModel)
    assert isinstance(layer1, ISAModel)
    X = T.TensorType('floatX', (False,) * 5)('x')
    layer1.in_chl = layer0.out_chl
    Y_l0 = layer0.fprop_conv(X, stride=LAYER0_STRIDE)
    Y_l1 = layer1.fprop_conv(Y_l0)
    with timed_operation('compiling fprop'):
        f = theano.function([X], Y_l1)

    def fprop(x):
        if x.ndim == 3:
            x = x.reshape(1, *x.shape)
        add_batch = False
        if x.ndim == 4:
            add_batch = True
            x = x.reshape(1, *x.shape)

        oshape = list(x.shape)
        oshape[1] = layer1.out_chl
        for i in range(2, 5):
            oshape[i] -= LAYER1_PATCH_SIZE - 1
        y_rst = np.empty(oshape, dtype=x.dtype)
        y_rst[:] = np.NAN

        prog = ProgressReporter('fprop', LAYER0_STRIDE ** 3)
        for i,j, k in itertools.product(range(LAYER0_STRIDE), repeat=3):
            y = f(x[:, :, i:, j:, k:])
            y_rst[:, :,
                  i::LAYER0_STRIDE, j::LAYER0_STRIDE, k::LAYER0_STRIDE] = y
            prog.trigger()
        prog.finish()
        y = y_rst
        assert np.isfinite(np.sum(y))

        if add_batch:
            assert y.shape[0] == 1
            y = y.reshape(y.shape[1:])
        return y
    return fprop

def main():
    parser = argparse.ArgumentParser(
        description='convert ISA model to convolution and apply to data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--l0', required=True, help='layer0 model')
    parser.add_argument('--l1', required=True, help='layer1 model')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    fprop = make_fprop(serial.load(args.l0), serial.load(args.l1))
    x = serial.load(args.input)
    logger.info('input shape: {}'.format(x.shape))
    y = fprop(x)
    logger.info('output shape: {}'.format(y.shape))
    opath = args.output
    opath = opath[:opath.rfind('.')] + '.pkl'
    serial.dump(y, opath)

if __name__ == '__main__':
    main()
