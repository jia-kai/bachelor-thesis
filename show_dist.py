#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: show_dist.py
# $Date: Sun May 03 17:32:23 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.visualize import view_3d_data_single

import numpy as np
import cv2
import matplotlib.pyplot as plt

import argparse
import logging
logger = logging.getLogger(__name__)

class ShowDist(object):
    _ftr0 = None
    _ftr1 = None
    _ftr1_sub = None
    _ftr1_slice = None
    _img_shape = None
    _conv_shape = None
    _dist_measure = None

    def __init__(self, img0, ftr0, img1, ftr1, dist_measure):
        self._dist_measure = dist_measure
        logger.info('img shape: {}; feature shape: {}'.format(
            img0.shape, ftr0.shape))
        assert img0.shape == img1.shape and ftr0.shape == ftr1.shape
        assert ftr0.ndim == 4 and img0.ndim == 3

        conv_shape = None
        for i in range(3):
            s = img0.shape[i] + 1 - ftr0.shape[i + 1]
            if conv_shape is None:
                conv_shape = s
            else:
                assert conv_shape == s
        self._conv_shape = conv_shape

        self._img_shape = np.array(img0.shape)
        self._ftr0 = ftr0
        self._ftr1 = ftr1
        view_3d_data_single(img0, onclick=self._on_img0_click, waitkey=False,
                            prefix='img0_')
        view_3d_data_single(img1, onaxischange=self._on_img1_axis_change,
                            waitkey=False, prefix='img1_')
        while True:
            key = chr(cv2.waitKey(-1) & 0xFF)
            if key == 'q':
                break


    def _on_img0_click(self, x, y, z):
        logger.info('click at {}, slice={}'.format(
            (x, y, z), self._ftr1_slice))
        if self._ftr1_sub is None:
            logger.warn('no suitable slice')
            return
        border = self._conv_shape / 2
        min_dist = min(x, y, z, np.min(self._img_shape - [x, y, z]))
        if min_dist < border:
            logger.warn('click out of feature boundary: min_dist={}'.format(
                min_dist))
            return
        x -= border
        y -= border
        z -= border
        ftr0 = self._ftr0[:, x, y, z].reshape(-1, 1, 1)
        rst = self._get_dist(ftr0, self._ftr1_sub)
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        matshow = ax.matshow(rst, interpolation='nearest')
        plt.colorbar(matshow)
        fig.show()

    def _get_dist(self, ref, ftrmap):
        if self._dist_measure.startswith('l'):
            norm = float(self._dist_measure[1:])
            logger.info('using dist: L{}'.format(norm))
            return np.power(
                np.power(np.abs(ref - ftrmap), norm).sum(axis=0),
                1.0 / norm)

        if self._dist_measure == 'sparse':
            logger.info('using dist: sparse')
            return 1 - ((ref > 0) == (ftrmap > 0)).mean(axis=0)

        if self._dist_measure == 'cos':
            logger.info('using dist: cos')
            norm_coeff = lambda v: 1.0 / np.sqrt(np.square(v).sum(
                axis=0, keepdims=True))
            ref = ref * norm_coeff(ref)
            ftrmap = ftrmap * norm_coeff(ftrmap)
            return 1 - (ref * ftrmap).sum(axis=0)

        raise RuntimeError('unknown dist: {}'.format(self._dist_measure))

    def _on_img1_axis_change(self, axis, pos):
        border = self._conv_shape / 2
        if min(pos, self._img_shape[axis] - pos) < border:
            self._ftr1_sub = None
            return
        pos -= border

        ind = [slice(None)] * 4
        ind[axis + 1] = pos
        self._ftr1_sub = self._ftr1[tuple(ind)]
        self._ftr1_slice = (axis, pos)


def main():
    parser = argparse.ArgumentParser(
        description='show distance of a point to each point on a slice',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('pack0')
    parser.add_argument('pack1')
    parser.add_argument('--measure', default='l2',
                        help='distance measure')
    args = parser.parse_args()

    pack0 = serial.load(args.pack0)
    pack1 = serial.load(args.pack1)
    ShowDist(pack0['img'], pack0['ftr'],
             pack1['img'], pack1['ftr'],
             args.measure)

if __name__ == '__main__':
    main()
