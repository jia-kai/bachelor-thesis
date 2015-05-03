#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: ftr_pca.py
# $Date: Sun May 03 16:47:01 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import argparse

from nasmia.math.ISA import ISA, ISAParam
from nasmia.utils import serial

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

    ftr = serial.load(args.input)
    param = ISAParam(in_dim=ftr.shape[1], subspace_size=1, hid_dim=args.dim)
    isa = ISA(param, ftr.T, args.nr_worker, map(int, args.gpus.split(',')))
    mdl = isa.get_model_pcaonly()
    serial.dump(
        {'W': mdl.coeff,
         'b': mdl.bias},
        args.output)

if __name__ == '__main__':
    main()
