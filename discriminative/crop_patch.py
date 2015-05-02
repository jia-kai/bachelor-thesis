#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: crop_patch.py
# $Date: Sat May 02 00:12:46 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial, ProgressReporter
from nasmia.utils.patch_cropper import CropPatchHelper

import numpy as np

import argparse
import os
import logging
logger = logging.getLogger(__name__)

def crop(args):
    osize = int(args.patch_size * args.bg_scale)
    if (osize - args.patch_size) % 2:
        osize += 1

    helper = CropPatchHelper(
        args.patch_size, (osize - args.patch_size) / 2,
        rng=np.random.RandomState(args.seed))

    basedir = os.path.dirname(args.imglist)
    with open(args.imglist) as fin:
        flist = fin.readlines()

    output = np.empty(
        (len(flist) * args.patch_per_img, osize, osize, osize),
        dtype='float32')
    logger.info('data size: {:.3f} GiB'.format(
        output.size * 4 / 1024.0**3))

    prog = ProgressReporter('crop', output.shape[0])

    output_idx = 0
    mask = None
    for fpath in flist:
        fpath = fpath.strip()
        data = serial.load(os.path.join(basedir, fpath))
        if args.masked:
            mask = serial.load(os.path.join(
                basedir, fpath.replace('orig', 'mask')))
        it = helper(data, mask)
        for i in range(args.patch_per_img):
            output[output_idx] = next(it)
            output_idx += 1
            prog.trigger()
    prog.finish()

    serial.dump(output, args.output)


def main():
    parser = argparse.ArgumentParser(
        description='randomly crop patches from nrrd image files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--patch_size', type=int, default=14)
    parser.add_argument('--bg_scale', type=float, default=1.5)
    parser.add_argument('--patch_per_img', type=int, default=250)
    parser.add_argument('--seed', type=int, default=20150501,
                        help='rng seed')
    parser.add_argument('--masked', action='store_true',
                        help='only crop patches within mask')
    parser.add_argument('imglist', help='path to list of images')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args()

    crop(args)

if __name__ == '__main__':
    main()
