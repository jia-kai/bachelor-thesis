#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: calc_border_dist.py
# $Date: Tue May 12 16:09:20 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import pyximport
pyximport.install()

from calc_border_dist_impl import calc_border_dist

from nasmia.utils import serial, timed_operation

import numpy as np

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='calculate distance to mask border',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='input mask image')
    parser.add_argument('output', help='border dist output')
    args = parser.parse_args()

    inp = serial.load(args.input)
    mask = (inp >= (inp.max() / 2)).astype(np.int32)
    with timed_operation('calc_border_dist: {}'.format(mask.shape)):
        output = calc_border_dist(mask)
    serial.dump(output, args.output)

if __name__ == '__main__':
    main()
