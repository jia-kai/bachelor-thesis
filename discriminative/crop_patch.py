#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: crop_patch.py
# $Date: Tue May 12 20:22:38 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial, ProgressReporter
from nasmia.utils.patch_cropper import CropPatchHelper
from nasmia.io import TrainingData
from nasmia.math.ISA.config import LAYER1_PATCH_SIZE

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

    data_shape = (len(flist) * args.patch_per_img, osize, osize, osize)
    logger.info('data: shape={} size={:.3f}GiB'.format(
        data_shape, np.prod(data_shape) * 4 / 1024.0**3))
    output = np.empty(data_shape, dtype='float32')

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

    data = TrainingData(patch=output, args=args)
    serial.dump(data, args.output)


def main():
    parser = argparse.ArgumentParser(
        description='randomly crop patches from nrrd image files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--patch_size', type=int, default=LAYER1_PATCH_SIZE)
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
