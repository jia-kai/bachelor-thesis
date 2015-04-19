#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: pack_img_ftr.py
# $Date: Sun Apr 19 23:42:30 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='pack image and feature pairs, '
        'possibly with ROI selection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('img')
    parser.add_argument('feature')
    parser.add_argument('output')
    parser.add_argument('--xmin', type=int, default=0)
    parser.add_argument('--xmax', type=int)
    parser.add_argument('--ymin', type=int, default=0)
    parser.add_argument('--ymax', type=int)
    parser.add_argument('--zmin', type=int, default=0)
    parser.add_argument('--zmax', type=int)
    args = parser.parse_args()

    img = serial.load(args.img)
    ftr = serial.load(args.feature)
    assert img.ndim == 3 and ftr.ndim == 4
    if args.xmax is None:
        args.xmax = img.shape[0]
    if args.ymax is None:
        args.ymax = img.shape[1]
    if args.zmax is None:
        args.zmax = img.shape[2]
    ftr = ftr[:,
              args.xmin:args.xmax + ftr.shape[1] - img.shape[0],
              args.ymin:args.ymax + ftr.shape[2] - img.shape[1],
              args.zmin:args.zmax + ftr.shape[3] - img.shape[2]]
    img = img[args.xmin:args.xmax, args.ymin:args.ymax, args.zmin:args.zmax]
    serial.dump({'img': img, 'ftr': ftr}, args.output)

if __name__ == '__main__':
    main()
