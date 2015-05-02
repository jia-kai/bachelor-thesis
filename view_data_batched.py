#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: view_data_batched.py
# $Date: Sat May 02 11:43:30 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.visualize import view_3d_data_batched

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='view a batch of 3D images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--channel', type=int,
                        help='select data channel')
    parser.add_argument('img_fpath')
    args = parser.parse_args()

    data = serial.load(args.img_fpath)
    print data.shape, data.dtype
    if args.channel is not None:
        assert data.ndim == 5
        data = data[:, args.channel]
    assert data.ndim == 4
    view_3d_data_batched(data)

if __name__ == '__main__':
    main()
