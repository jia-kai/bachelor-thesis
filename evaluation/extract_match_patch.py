#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: extract_match_patch.py
# $Date: Wed Jun 10 23:36:38 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.io import PointMatchResult

import numpy as np

import random
import argparse
import os
import logging
from itertools import izip_longest
logger = logging.getLogger(__name__)

class PatchLoader(object):
    _args = None
    _img_cache = None
    _border_cache = None

    _write_idx = 0
    _write_idx_seq = None
    _result = None
    _result_dist = None

    def __init__(self, args):
        self._args = args
        self._result = self._gen_empty_result()
        self._result_dist = self._gen_empty_result()
        self._img_cache = {}
        self._border_cache = {}

    def _gen_empty_result(self):
        sz = self._args.patch_radius * 2 + 1
        return np.zeros((self._args.nr_keep, 2, 3, sz, sz, sz),
                        dtype=np.float32)

    def _next_write_idx(self):
        rst = self._write_idx
        self._write_idx += 1
        if rst < self._result.shape[0]:
            return rst

        if not self._write_idx_seq:
            self._write_idx_seq = range(self._result.shape[0])
            random.shuffle(self._write_idx_seq)

        return self._write_idx_seq.pop()


    def _load_img(self, num):
        if num in self._img_cache:
            return self._img_cache[num]
        img = serial.load(
            os.path.join(self._args.img_dir, '{}.nii.gz'.format(num)),
            np.ndarray)
        self._img_cache[num] = img
        return img

    def _load_border(self, num):
        if num in self._border_cache:
            return self._border_cache[num]
        img = serial.load(
            os.path.join(self._args.border_dir, '{}.nii.gz'.format(num)),
            np.ndarray)
        self._border_cache[num] = img
        return img

    def feed(self, fname):
        pt_match = serial.load(fname)
        assert isinstance(pt_match, PointMatchResult)
        ref_num, test_num = os.path.basename(fname).split('.')[0].split('-')
        ref_img, test_img = map(self._load_img, (ref_num, test_num))
        ref_bd, test_bd = map(self._load_border, (ref_num, test_num))

        ref_dist = pt_match.args.ref_dist
        min_geo_dist = self._args.dist_min
        max_geo_dist = self._args.dist_max
        assert pt_match.img_shape == test_img.shape

        r = self._args.patch_radius

        for i in range(pt_match.ref_idx.shape[0]):
            x1, y1, z1 = pt_match.idx[i]
            geo_dist = test_bd[x1, y1, z1] - ref_dist
            if geo_dist < min_geo_dist or geo_dist > max_geo_dist:
                continue
            x0, y0, z0 = pt_match.ref_idx[i]
            p0 = ref_img[x0-r:x0+r+1, y0-r:y0+r+1, z0-r:z0+r+1]
            p1 = test_img[x1-r:x1+r+1, y1-r:y1+r+1, z1-r:z1+r+1]
            if min(min(p0.shape), min(p1.shape)) != r * 2 + 1:
                continue

            if random.random() * self._write_idx <= self._result.shape[0]:
                idx = self._next_write_idx()
                dest = self._result[idx]
                vmin = min(p0.min(), p1.min())
                vmax = max(p0.max(), p1.max())
                dest[0] = (p0 - vmin) / (vmax - vmin + 1e-9) * 255.0
                dest[1] = (p1 - vmin) / (vmax - vmin + 1e-9) * 255.0

                dest = self._result_dist[idx]
                dest[0] = ref_bd[x0-r:x0+r+1, y0-r:y0+r+1, z0-r:z0+r+1]
                dest[1] = test_bd[x1-r:x1+r+1, y1-r:y1+r+1, z1-r:z1+r+1]

    def get_result(self):
        r = self._args.patch_radius
        sz = r * 2 + 1
        rst = self._result.reshape(
            self._result.shape[0] * 2, 3, sz, sz, sz).copy()

        rhl = self._args.highlight_radius
        assert r > rhl
        hl_range = (slice(None), ) + (slice(r - rhl, rhl - r), ) * 3

        rst[:, 0][hl_range] = 0
        rst[:, 2][hl_range] = 0
        return rst

    def get_result_dist(self):
        r = self._args.patch_radius
        sz = r * 2 + 1
        return -np.abs(self._result_dist.reshape(
            self._result.shape[0] * 2, 3, sz, sz, sz))


def main():
    parser = argparse.ArgumentParser(
        description='extract matched patches',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--border_dir',
                        default=os.path.abspath(os.path.join(
                            os.path.dirname(__file__),
                            '../../sliver07/border-dist')),
                        help='directory that store the border dist files')
    parser.add_argument('--img_dir',
                        default=os.path.abspath(os.path.join(
                            os.path.dirname(__file__),
                            '../../sliver07/train-cropped')),
                        help='directory that store original image files')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--out_bd_dist', help='output border dist')
    parser.add_argument('--dist_min', type=int, default=-1,
                        help='minimum dist to be considered as hit')
    parser.add_argument('--dist_max', type=int, default=1,
                        help='maximum dist to be considered as hit')
    parser.add_argument('--patch_radius', type=int, default=20,
                        help='radius for patch to be extracted')
    parser.add_argument('--highlight_radius', type=int, default=10,
                        help='radius for patch to be extracted')
    parser.add_argument('--nr_keep', type=int, default=100,
                        help='number of results to be kept')
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()

    loader = PatchLoader(args)
    for i in args.input:
        logger.info('loading {}'.format(i))
        loader.feed(i)
    logger.info('write result to {}'.format(args.output))
    serial.dump(loader.get_result(), args.output)
    if args.out_bd_dist:
        logger.info('write border dist to {}'.format(args.out_bd_dist))
        serial.dump(loader.get_result_dist(), args.out_bd_dist)

if __name__ == '__main__':
    main()
