#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: apply_mask.py
# $Date: Fri May 01 23:27:13 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.visualize import view_3d_data_single

import argparse

import nibabel as nib
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description='apply mask to an image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('img')
    parser.add_argument('mask')
    parser.add_argument('output')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    img = serial.load(args.img)
    mask = serial.load(args.mask)
    img[mask <= mask.max() / 2.0] = img.min()
    if args.verbose:
        view_3d_data_single(img)

    out = nib.Nifti1Pair(img, np.eye(4))
    nib.save(out, args.output)

if __name__ == '__main__':
    main()
