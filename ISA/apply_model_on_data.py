#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: conv_model_to_data.py
# $Date: Sun Apr 05 15:55:18 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial, timed_operation
from nasmia.utils.conv_splitter import ConvSplitter

import theano.tensor as T
import theano
import numpy as np

import logging
import argparse
import os
logger = logging.getLogger(__name__)

def read_file_list(flist):
    basedir = os.path.dirname(flist)
    with open(flist) as fin:
        return [os.path.join(basedir, i.strip())
                for i in fin]

def make_fprop(model_fpath, split):
    model = serial.load(model_fpath)
    X = T.TensorType('floatX', (False,) * 5)('x')
    Y = model.fprop_conv(X)
    with timed_operation('compiling fprop'):
        f = theano.function([X], Y)

    f_split = ConvSplitter(
        2, 5, split, model.conv_kern_shape, model.out_chl, f)
    f_split.progress = True

    def fprop(x):
        if x.ndim == 3:
            x = x.reshape(1, *x.shape)
        add_batch = False
        if x.ndim == 4:
            add_batch = True
            x = x.reshape(1, *x.shape)
        y = f_split(x)
        if add_batch:
            assert y.shape[0] == 1
            y = y.reshape(y.shape[1:])
        return y
    return fprop

def main():
    parser = argparse.ArgumentParser(
        description='convert ISA model to convolution and apply to data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--split', default='1',
                        help='split data before fprop to fit in GPU memory')
    parser.add_argument('model')
    parser.add_argument('input_list')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    
    fprop = make_fprop(args.model, map(int, args.split.split(',')))
    inp_list = read_file_list(args.input_list)

    for fpath in inp_list:
        x = serial.load(fpath)
        logger.info('working on {}, shape={}'.format(fpath, x.shape))
        y = fprop(x)
        opath = os.path.join(args.output_dir, os.path.basename(fpath))
        opath = opath[:opath.rfind('.')] + '.pkl'
        serial.dump(y, opath)

if __name__ == '__main__':
    main()
