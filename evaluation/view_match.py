#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: view_match.py
# $Date: Sun May 31 16:33:12 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.visualize import view_3d_data_single
from nasmia.io import PointMatchResult
from nasmia.utils import serial

import numpy as np

import argparse
import logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='view point match result',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', nargs='+')
    parser.add_argument('--radius', type=int, default=2,
                        help='radius of points drawn on image')
    parser.add_argument('--mask', help='plot groundtruth mask')
    parser.add_argument('--dist_thresh', type=float, default=float('inf'),
                        help='feature dist threshold')
    args = parser.parse_args()

    img = None
    nr_pt_view = 0
    for i in args.input:
        logger.info('load {}'.format(i))
        pmtch_rst = serial.load(i, PointMatchResult)
        img_shape = pmtch_rst.img_shape
        if args.mask:
            img_shape = (3, ) + img_shape
        if img is None:
            img = np.zeros(img_shape)
            if args.mask:
                mask = serial.load(args.mask, np.ndarray)
                mask = mask != 0
                img[:] = np.expand_dims(mask, axis=0)
        else:
            assert img.shape == img_shape

        dist = 1 / (pmtch_rst.dist + 1e-5)
        dist += dist.max() / 2
        dist *= 1.0 / dist.max()

        r = args.radius
        for k in range(pmtch_rst.dist.shape[0]):
            if pmtch_rst.dist[k] > args.dist_thresh:
                continue
            nr_pt_view += 1
            val = dist[k]
            x, y, z = pmtch_rst.idx[k]
            x0, y0, z0 = [max(i - r, 0) for i in (x, y, z)]
            x1, y1, z1 = [i + r for i in (x, y, z)]
            if args.mask:
                img[0, x0:x1, y0:y1, z0:z1] = 0
                img[1, x0:x1, y0:y1, z0:z1] = val
                img[2, x0:x1, y0:y1, z0:z1] = val
            else:
                img[x0:x1, y0:y1, z0:z1] = val

    logger.info('number of points: {}'.format(nr_pt_view))
    view_3d_data_single(img)

if __name__ == '__main__':
    main()
