#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: view_data_batched.py
# $Date: Tue Jun 09 15:49:03 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.visualize import view_3d_data_batched

import numpy as np

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
    parser.add_argument('--sliced', action='store_true',
                        help='view sliced data of a single image')
    parser.add_argument('--max_win_width', type=int, default=600,
                        help='max display window width')
    parser.add_argument('--max_win_height', type=int, default=600,
                        help='max display window height')
    parser.add_argument('img_fpath', nargs='+',
                        help='path to image; if multiple paths are given, they'
                        ' would be merged')
    args = parser.parse_args()

    data = serial.load(args.img_fpath[0], np.ndarray)
    if len(args.img_fpath) > 1:
        all_data = [data]
        for i in args.img_fpath[1:]:
            cur_data = serial.load(i, np.ndarray)
            assert cur_data.shape == data.shape, (
                data.shape, cur_data.shape)
            all_data.append(cur_data)
        data = np.array(all_data)
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
    if args.sliced:
        assert data.ndim <= 4
        if data.ndim  == 4:
            data = np.expand_dims(data, axis=2)
            data = np.swapaxes(data, 0, 1)
            data = data[64::4]
        else:
            assert data.ndim == 3
            data = np.expand_dims(data, axis=1)
    view_3d_data_batched(data, max_width=args.max_win_width,
                         max_height=args.max_win_height)

if __name__ == '__main__':
    main()
