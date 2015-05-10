#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: crop_patch_l0.py
# $Date: Sun May 10 15:41:27 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial, ProgressReporter
from nasmia.utils.patch_cropper import CropPatchHelper

import numpy as np

import argparse
import os
import logging
logger = logging.getLogger(__name__)

def crop(args):
    basedir = os.path.dirname(args.imglist)
    with open(args.imglist) as fin:
        flist = fin.readlines()

    helper = CropPatchHelper(
        args.patch_size, 0, rng=np.random.RandomState(args.seed))

    output = np.empty(
        (args.patch_size**3, len(flist) * args.patch_per_img),
        dtype='float32')
    logger.info('data size: {:.3f} GiB'.format(
        output.size * 4 / 1024.0**3))

    prog = ProgressReporter('crop', output.shape[1])

    output_idx = 0
    for fpath in flist:
        fpath = fpath.strip()
        data = serial.load(os.path.join(basedir, fpath))
        it = helper(data)
        for i in range(args.patch_per_img):
            output[:, output_idx] = next(it).flatten()
            output_idx += 1
            prog.trigger()
    prog.finish()

    serial.dump(output, args.output)


def main():
    parser = argparse.ArgumentParser(
        description='randomly crop patches from nrrd image files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--patch_size', type=int, default=13)
    parser.add_argument('--patch_per_img', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=20150328,
                        help='rng seed')
    parser.add_argument('imglist', help='path to list of images')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args()

    crop(args)

if __name__ == '__main__':
    main()
