#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: dump_model_filter.py
# $Date: Sun May 31 21:03:03 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.math.ISA.model import ISAModel
from nasmia.utils import serial

import numpy as np

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='visualize model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model')
    parser.add_argument('output')
    parser.add_argument('--in_chl', type=int,
                        help='specify model input channels')
    args = parser.parse_args()

    model = serial.load(args.model, ISAModel)
    if args.in_chl:
        model.in_chl = args.in_chl
    data = model.get_conv_coeff()
    serial.dump(data, args.output, use_pickle=True)

if __name__ == '__main__':
    main()
