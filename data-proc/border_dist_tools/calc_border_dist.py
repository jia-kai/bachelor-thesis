#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: calc_border_dist.py
# $Date: Mon May 11 19:48:28 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import pyximport
pyximport.install()

from calc_border_dist_impl import calc_border_dist

from nasmia.utils import serial

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
    mask = (inp >= (inp.max() / 2)).astype(np.int)
    output = calc_border_dist(mask)
    serial.dump(output, args.output)

if __name__ == '__main__':
    main()
