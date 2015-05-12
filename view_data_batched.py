#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: view_data_batched.py
# $Date: Tue May 12 20:56:10 2015 +0800
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
    parser.add_argument('--unfold_isa', action='store_true',
                        help='unfold data used for ISA training')
    parser.add_argument('--unfold_chl', action='store_true',
                        help='unfold data with multiple channels')
    parser.add_argument('img_fpath')
    args = parser.parse_args()

    data = serial.load(args.img_fpath)
    print data.shape, data.dtype
    if args.unfold_isa:
        assert data.ndim == 2
        shp = int(data.shape[0] ** (1.0 / 3) + 0.5)
        data = data.T.reshape(-1, shp, shp, shp)
    elif args.unfold_chl:
        assert data.ndim == 5
        data = data.reshape((-1, ) + data.shape[2:])
    elif args.channel is not None:
        assert data.ndim == 5
        data = data[:, args.channel]
    assert data.ndim == 4
    view_3d_data_batched(data)

if __name__ == '__main__':
    main()
