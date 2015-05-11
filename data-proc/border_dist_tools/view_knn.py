#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: view_knn.py
# $Date: Mon May 11 22:08:56 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.visualize import view_3d_data_single
from nasmia.io import KNNResult
from nasmia.utils import serial

import numpy as np

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='view KNN result',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input')
    parser.add_argument('--radius', type=int, default=4)
    args = parser.parse_args()

    knn = serial.load(args.input, KNNResult)
    img = np.zeros(knn.img_shape)
    r = args.radius
    for ptnum in range(knn.dist.shape[0]):
        for k in range(knn.dist.shape[1]):
            x, y, z = knn.idx[ptnum, k]
            img[max(x-r, 0):x+r, max(y-r, 0):y+r, max(z-r, 0):z+r] = (
                knn.dist[ptnum, k])

    view_3d_data_single(img)

if __name__ == '__main__':
    main()
