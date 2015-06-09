#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: view_match.py
# $Date: Mon Jun 08 10:12:50 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.visualize import view_3d_data_single
from nasmia.io import PointMatchResult
from nasmia.utils import serial

import numpy as np

import argparse
import logging
logger = logging.getLogger(__name__)

def load_image(args, img_shape):
    img = np.zeros(img_shape)
    if args.orig:
        orig = serial.load(args.orig, np.ndarray)
        img[:] = np.expand_dims(orig, axis=0)
        img -= img.min()

    if args.mask:
        mask = serial.load(args.mask, np.ndarray)
        if args.orig:
            img[0][mask != 0] *= 0.6
            img[1][mask != 0] *= 0.8
        else:
            img[0, :] = mask != 0
    return img

def main():
    parser = argparse.ArgumentParser(
        description='view point match result',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', nargs='+')
    parser.add_argument('--radius', type=int, default=2,
                        help='radius of points drawn on image')
    parser.add_argument('--mask', help='plot groundtruth mask')
    parser.add_argument('--orig', help='plot original test image')
    parser.add_argument('--is_roc_dist_dump', action='store_true',
                        help='input file is ROC dist dump')
    parser.add_argument('--border_dist_min', type=int, default=-1,
                        help='min border dist to be considered as matched')
    parser.add_argument('--border_dist_max', type=int, default=1,
                        help='max border dist to be considered as matched')
    parser.add_argument('--dist_thresh', type=float, default=float('inf'),
                        help='feature dist threshold')
    parser.add_argument('-o', '--output', help='write final image to file')
    args = parser.parse_args()

    img = None
    nr_pt_view = 0
    if args.is_roc_dist_dump:
        assert len(args.input) == 1
        nr_matched = 0
        tot_nr_point = 0

    for i in args.input:
        logger.info('load {}'.format(i))
        pmtch_rst = serial.load(i, PointMatchResult)
        img_shape = pmtch_rst.img_shape
        if args.mask:
            img_shape = (3, ) + img_shape
        if img is None:
            img = load_image(args, img_shape)
            #view_3d_data_single(img)
        else:
            assert img.shape == img_shape

        dist = -np.log(pmtch_rst.dist + 1e-5)
        dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-20) * 0.5
        dist += 0.5
        dist *= img.max()

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
                color = np.zeros([3, 1, 1, 1], dtype=img.dtype)
                color[0] = val
                if args.is_roc_dist_dump:
                    tot_nr_point += 1
                    geo_dist = pmtch_rst.geo_dist[k]
                    if (geo_dist >= args.border_dist_min and
                            geo_dist <= args.border_dist_max):
                        nr_matched += 1
                        color[:, 0, 0, 0] = (0, val, 0)
                img[:, x0:x1, y0:y1, z0:z1] = color
            else:
                img[x0:x1, y0:y1, z0:z1] = val

    logger.info('number of points: {}'.format(nr_pt_view))
    if args.is_roc_dist_dump:
        logger.info('match accuracy: {}; top ratio: {}'.format(
            float(nr_matched) / tot_nr_point,
            float(tot_nr_point) / pmtch_rst.dist.shape[0]))
    if args.output:
        serial.dump(img, args.output, use_pickle=True)
    else:
        view_3d_data_single(img)

if __name__ == '__main__':
    main()
