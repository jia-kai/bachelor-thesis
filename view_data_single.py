#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: view_data_single.py
# $Date: Mon May 11 20:08:44 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.visualize import view_3d_data_single

import numpy as np
import cv2

import argparse
import logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='view an 3D image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--channel', type=int,
                        help='select data channel')
    parser.add_argument('--vselect', type=float, help='select value to disp')
    parser.add_argument('img_fpath')
    args = parser.parse_args()

    data = serial.load(args.img_fpath)
    print data.shape, data.dtype
    if args.channel is not None:
        assert data.ndim == 4
        data = data[args.channel]
    assert data.ndim == 3
    if args.vselect is not None:
        data = (data == args.vselect)
        logger.info('nr points selected: {}'.format(data.sum()))
    view_3d_data_single(data)

if __name__ == '__main__':
    main()
