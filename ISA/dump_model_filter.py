#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: dump_model_filter.py
# $Date: Sun May 10 21:34:27 2015 +0800
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
    args = parser.parse_args()

    model = serial.load(args.model, ISAModel)
    patch_size = int(model.coeff.shape[1] ** (1.0/3) + 0.5)
    assert patch_size ** 3 == model.coeff.shape[1]

    data = model.coeff
    data = data.reshape(data.shape[0], patch_size, patch_size, patch_size)
    serial.dump(data, args.output, use_pickle=True)

if __name__ == '__main__':
    main()
