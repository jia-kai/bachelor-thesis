#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: view_model.py
# $Date: Sun Apr 05 11:59:33 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.math.ISA.model import ISAModel
from nasmia.utils import serial
from nasmia.visualize import view_3d_data_simple

import cv2
import numpy as np

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='visualize model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sep', type=int, default=1,
                        help='space between adjacent patches')
    parser.add_argument('-o', '--output', help='filter output file')
    parser.add_argument('model')
    args = parser.parse_args()

    model = serial.load(args.model, ISAModel)
    patch_size = int(model.coeff.shape[1] ** (1.0/3) + 0.5)
    assert patch_size ** 3 == model.coeff.shape[1]
    patch_shape = (patch_size, patch_size, patch_size)

    #view_3d_data_simple(-model.bias.reshape(patch_shape), 10)
    #view_3d_data_simple(model.coeff[3].reshape(patch_shape), 10)

    data = model.coeff
    data = data.reshape(data.shape[0], patch_size, patch_size, patch_size)

    data = data[:, :, :, patch_size / 2]
    nr = int(data.shape[0] ** 0.5)
    nc = data.shape[0] / nr
    data = data[:nr*nc].reshape(nr, nc, patch_size, patch_size)

    dispsize = patch_size + args.sep
    img = np.zeros((dispsize * data.shape[0] - args.sep,
                    dispsize * data.shape[1] - args.sep),
                   dtype='uint8')
    row = 0
    for prow in data:
        col = 0
        for pcol in prow:
            dmin, dmax = pcol.min(), pcol.max()
            pcol = (pcol - dmin) * (255.0 / (dmax - dmin))
            img[row:row+patch_size, col:col+patch_size] = pcol
            col += dispsize
        row += dispsize

    if args.output:
        cv2.imwrite(args.output, img)
    cv2.imshow('img', img)
    cv2.waitKey(-1)

if __name__ == '__main__':
    main()
