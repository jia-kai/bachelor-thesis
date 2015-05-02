#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: crop_img.py
# $Date: Sat May 02 20:15:04 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import argparse
from nasmia.utils import serial

def main():
    parser = argparse.ArgumentParser(
        description='crop 3D image at given position',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--xmin', type=int, default=0)
    parser.add_argument('--xmax', type=int)
    parser.add_argument('--ymin', type=int, default=0)
    parser.add_argument('--ymax', type=int)
    parser.add_argument('--zmin', type=int, default=0)
    parser.add_argument('--zmax', type=int)
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    img = serial.load(args.input)
    assert img.ndim == 3

    if args.xmax is None:
        args.xmax = img.shape[0]
    if args.ymax is None:
        args.ymax = img.shape[1]
    if args.zmax is None:
        args.zmax = img.shape[2]
    img = img[args.xmin:args.xmax, args.ymin:args.ymax, args.zmin:args.zmax]

    serial.dump(img, args.output)

if __name__ == '__main__':
    main()
