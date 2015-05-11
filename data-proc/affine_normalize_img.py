#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: affine_normalize_img.py
# $Date: Mon May 11 21:14:36 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.math.op.affine3d import batched_affine3d

import nibabel
import numpy as np

import argparse
import logging
logger = logging.getLogger(__name__)

def load_and_normalize(fpath, scale):
    ni_img = nibabel.load(fpath)
    k0, k1, k2, k3 = np.diag(ni_img.affine)
    assert k3 == 1 and k0 < 0 and k1 < 0 and k2 > 0
    assert np.all(ni_img.affine == np.diag([k0, k1, k2, k3]))

    k0, k1, k2 = k0 * scale, k1 * scale, k2 * scale

    img = ni_img.get_data()
    out_shape = [int(abs(i) + 0.5) for i in (
        k0 * img.shape[0], k1 * img.shape[1], k2 * img.shape[2])]

    affine_mat = np.zeros((3, 4))
    affine_mat[:, :3] = np.diag([1 / k0, 1 / k1, 1 / k2])
    affine_mat[:, 3] = img.shape[0], img.shape[1], 0
    affine_mat = np.expand_dims(affine_mat, axis=0)

    rst = batched_affine3d(
        np.expand_dims(img, axis=0), affine_mat, out_shape)
    return rst[0]


def apply_mask(image, mask, enlarge, min_border):
    mask = (mask >= mask.max() / 2).astype('float32')
    subidx = []
    for axis in range(image.ndim):
        def visit(v):
            idx = [slice(None)] * image.ndim
            idx[axis] = v
            tup = tuple(idx)
            return mask[tup].sum()
        low = 0
        while not visit(low):
            low += 1
        high = image.shape[axis] - 1
        while not visit(high):
            high -= 1
        assert low < high
        low0, high0 = low, high
        delta = int(max((high - low) * (enlarge - 1) / 2, min_border))
        low = max(0, low - delta)
        high = min(image.shape[axis], high + delta)
        subidx.append(slice(low, high + 1))

    return image[subidx], mask[subidx]


def main():
    parser = argparse.ArgumentParser(
        description='apply affine transform stored '
        'with image to normalize image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image', required=True,
                        help='input image')
    parser.add_argument('-m', '--mask',
                        help='mask image; crop input if provided')
    parser.add_argument('-s', '--scale', type=float, default=1,
                        help='scale of output image')
    parser.add_argument('--mask_crop_enlarge', type=float, default=1.2,
                        help='enlarge factor for mask crop box')
    parser.add_argument('--min_mask_border', type=int, default=0,
                        help='minimum distance from mask to border')
    parser.add_argument('-o', '--output', required=True,
                        help='output filename; do not include extension')
    args = parser.parse_args()

    image = load_and_normalize(args.image, args.scale)
    if args.mask:
        mask = load_and_normalize(args.mask, args.scale)
        assert image.shape == mask.shape
        image, mask = apply_mask(
            image, mask, args.mask_crop_enlarge, args.min_mask_border)
        mask = nibabel.Nifti1Pair(mask, np.eye(4))


    image = nibabel.Nifti1Pair(image, np.eye(4))
    logger.info('output size: {}'.format(image.shape))
    nibabel.save(image, args.output + '.nii.gz')
    if args.mask:
        nibabel.save(mask, args.output + '-mask.nii.gz')

if __name__ ==  '__main__':
    main()
