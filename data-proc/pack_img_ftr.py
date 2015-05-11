#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: pack_img_ftr.py
# $Date: Sat May 02 20:15:03 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='pack image and feature pairs for show_dist',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('img')
    parser.add_argument('feature')
    parser.add_argument('output')
    args = parser.parse_args()

    img = serial.load(args.img)
    ftr = serial.load(args.feature)
    assert img.ndim == 3 and ftr.ndim == 4
    serial.dump({'img': img, 'ftr': ftr}, args.output)

if __name__ == '__main__':
    main()
