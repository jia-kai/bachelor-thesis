#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: ftr_pca.py
# $Date: Mon May 25 22:27:58 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.math.ISA import ISA, ISAParam
from nasmia.utils import serial

import numpy as np

import logging
import argparse
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='perform PCA on model features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('-d', '--dim', type=int, required=True,
                        help='PCA dimension')
    parser.add_argument('--nr_worker', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--gpus', default='0,1,2,3',
                        help='gpus to use')
    args = parser.parse_args()

    ftr = serial.load(args.input, np.ndarray)

    logger.info('input feature shape: {}'.format(ftr.shape))

    param = ISAParam(in_dim=ftr.shape[1], subspace_size=1, hid_dim=args.dim)
    isa = ISA(param, ftr.T, args.nr_worker, map(int, args.gpus.split(',')))
    mdl = isa.get_model_pcaonly()
    serial.dump(
        {'W': mdl.coeff,
         'b': mdl.bias},
        args.output)

if __name__ == '__main__':
    main()
