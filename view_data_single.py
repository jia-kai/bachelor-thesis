#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: view_data_single.py
# $Date: Fri May 01 23:28:31 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.visualize import view_3d_data_single

import numpy as np
import cv2

import argparse

def draw_surface_box_on_image(surface, image):
    tr = lambda p: ((p - image.origin) / image.spacing).astype('int')
    pmin = tr(np.min(surface.points, axis=0))
    pmax = tr(np.max(surface.points, axis=0))
    assert pmin.min() >= 0 and np.all(pmax <= image.data.shape)
    data = image.data
    vmax = data.max()
    data = data[:,  :, :, np.newaxis]
    data = np.concatenate((data, data, data), 3)
    color = (0, 0, vmax)
    x0, y0, z0 = pmin
    x1, y1, z1 = pmax
    data[x0, y0:y1, z0:z1] = color
    data[x1, y0:y1, z0:z1] = color
    data[x0:x1, y0, z0:z1] = color
    data[x0:x1, y1, z0:z1] = color
    data[x0:x1, y0:y1, z0] = color
    data[x0:x1, y0:y1, z1] = color
    return data

def main():
    parser = argparse.ArgumentParser(
        description='view an 3D image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--channel', type=int,
                        help='select data channel')
    parser.add_argument('img_fpath')
    args = parser.parse_args()

    data = serial.load(args.img_fpath)
    print data.shape, data.dtype
    if args.channel is not None:
        assert data.ndim == 4
        data = data[args.channel]
    assert data.ndim == 3
    view_3d_data_single(data)

if __name__ == '__main__':
    main()
