#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: show_dist.py
# $Date: Sun Apr 19 22:48:43 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.visualize import view_3d_data_simple
from nasmia.math.ISA.config import LAYER1_PATCH_SIZE

import numpy as np
import cv2
import matplotlib.pyplot as plt

import argparse
import logging
logger = logging.getLogger(__name__)

def normalize_features(ftr):
    return ftr / np.sqrt(np.sum(np.square(ftr), axis=0, keepdims=True))

class ShowDist(object):
    _ftr0 = None
    _ftr1 = None
    _ftr1_sub = None
    _ftr1_slice = None

    def __init__(self, img0, ftr0, img1, ftr1):
        logger.info('img shape: {}; feature shape: {}'.format(
            img0.shape, ftr0.shape))
        assert img0.shape == img1.shape and ftr0.shape == ftr1.shape
        assert ftr0.ndim == 4 and img0.ndim == 3
        for i in range(3):
            assert ftr0.shape[i + 1] == img0.shape[i] - LAYER1_PATCH_SIZE + 1
        self._ftr0 = normalize_features(ftr0)
        self._ftr1 = normalize_features(ftr1)
        view_3d_data_simple(img0, onclick=self._on_img0_click, waitkey=False,
                            prefix='img0_')
        view_3d_data_simple(img1, onaxischange=self._on_img1_axis_change,
                            waitkey=False, prefix='img1_')
        while True:
            key = chr(cv2.waitKey(-1) & 0xFF)
            if key == 'q':
                break


    def _on_img0_click(self, x, y, z):
        logger.info('click at {}, slice={}'.format(
            (x, y, z), self._ftr1_slice))
        x -= LAYER1_PATCH_SIZE / 2
        y -= LAYER1_PATCH_SIZE / 2
        z -= LAYER1_PATCH_SIZE / 2
        if min(x, y, z, self._ftr0.shape[1] - max(x, y, z)) < 0:
            logger.warn('click out of feature boundary')
            return
        ftr0 = self._ftr0[:, x, y, z].reshape(1, -1)
        rst = np.tensordot(ftr0, self._ftr1_sub, axes=(1, 0))[0]
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        matshow = ax.matshow(rst, interpolation='nearest')
        plt.colorbar(matshow)
        fig.show()

    def _on_img1_axis_change(self, axis, pos, ind):
        self._ftr1_sub = self._ftr1[tuple([slice(None)] + list(ind))]
        self._ftr1_slice = (axis, pos)


def main():
    parser = argparse.ArgumentParser(
        description='show distance of a point to each point on a slice',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('img0')
    parser.add_argument('ftr0')
    parser.add_argument('img1')
    parser.add_argument('ftr1')
    args = parser.parse_args()

    ShowDist(*map(serial.load, [args.img0, args.ftr0, args.img1, args.ftr1]))

if __name__ == '__main__':
    main()
