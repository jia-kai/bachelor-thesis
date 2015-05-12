#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: get_data_stat.py
# $Date: Tue May 12 20:30:12 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.io import TrainingData

import numpy as np

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    data = serial.load(args.input)
    assert isinstance(data, TrainingData)
    psize = data.args.patch_size
    border = data.patch.shape[1] - psize
    assert border % 2 == 0
    border /= 2

    patch = data.patch[:, border:-border, border:-border, border:-border]
    pmean = patch.mean()
    print 'mean:', pmean
    print 'max_abs:', np.abs(patch - pmean).max()

if __name__ == '__main__':
    main()
