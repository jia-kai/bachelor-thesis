#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: apply_model_on_data.py
# $Date: Fri Jun 12 12:22:39 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial, timed_operation, ProgressReporter
from nasmia.math.ISA.model import ISAModel
from nasmia.math.ISA.config import LAYER1_PATCH_SIZE, LAYER0_STRIDE
from nasmia.math.op import sharedX
from nasmia.io import ModelEvalOutput

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
        y_rst = np.empty(oshape, dtype='float32')
        y_rst[:] = np.NAN
        logger.info('total feature size: {:.2f}MiB'.format(
                    y_rst.size * 4 / 1024.0**2))

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

def calc_raw_output(inp, layer0, layer1):
    assert inp.ndim == 4 and all(i == LAYER1_PATCH_SIZE for i in inp.shape[1:])
    inp = np.expand_dims(inp, axis=1)
    hidv = layer0.fprop_conv(sharedX(inp), stride=LAYER0_STRIDE).eval()
    assert hidv.ndim == 5 and hidv.shape[2:] == (2, 2, 2)
    hidv = hidv.reshape(inp.shape[0], layer0.out_chl * 8)
    return layer1(hidv.T).T


def check_output(img, output, layer0, layer1, nr_check):
    mdl_shp = LAYER1_PATCH_SIZE
    inp = np.empty(shape=[nr_check] + [mdl_shp] * 3, dtype='float32')
    loc = []
    rand = lambda v: np.random.randint(0, img.shape[v] - mdl_shp + 1)
    for i in inp:
        x, y, z = map(rand, range(3))
        i[:] = img[x:x+mdl_shp, y:y+mdl_shp, z:z+mdl_shp]
        loc.append((x, y, z))

    expected_all = calc_raw_output(inp, layer0, layer1)

    max_err = 0
    max_loc = None
    for (x, y, z), expected in zip(loc, expected_all):
        got = output[:, x, y, z]
        err = np.abs(got - expected).max()
        if err > max_err:
            max_err = err
            max_loc = (x, y, z)
    logger.info('err: max={} loc={}'.format(max_err, max_loc))


def main():
    parser = argparse.ArgumentParser(
        description='convert ISA model to convolution and apply to data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--l0', required=True, help='layer0 model')
    parser.add_argument('--l1', required=True, help='layer1 model')
    parser.add_argument('--check', type=int,
                        help='number of patches to check fprop output')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    layer0, layer1 = map(serial.load, (args.l0, args.l1))
    fprop = make_fprop(layer0, layer1)
    x = serial.load(args.input)
    logger.info('input shape: {}'.format(x.shape))
    y = fprop(x)
    logger.info('output shape: {}'.format(y.shape))
    opath = args.output
    if '.' in opath and ('/' not in opath or '.' in opath[opath.rfind('/'):]):
        opath = opath[:opath.rfind('.')] + '.pkl'
    else:
        opath += '.pkl'
    serial.dump(ModelEvalOutput(img=x, ftr=y), opath)
    logger.info('result wrote to {}'.format(opath))

    if args.check:
        check_output(x, y, layer0, layer1, args.check)

if __name__ == '__main__':
    main()
