#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: main.py
# $Date: Sat Mar 21 16:40:23 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.thirdparty import nrrd
from nasmia.io import ScenePackReader
from nasmia.visualize import view_3d_data_simple

import numpy as np
import cv2

import argparse

def draw_surface_box_on_image(surface, image):
    tr = lambda p: ((p - image.origin) / image.spacing).astype('int')
    pmin = tr(np.min(surface.points, axis=0))
    pmax = tr(np.max(surface.points, axis=0))
    assert pmin.min() >= 0 and np.all(pmax <= image.data.shape)
    data = image.data
    vmax = data.max()
    data = data[:,  :, :, np.newaxis]
    data = np.concatenate((data, data, data), 3)
    color = (0, 0, vmax)
    x0, y0, z0 = pmin
    x1, y1, z1 = pmax
    data[x0, y0:y1, z0:z1] = color
    data[x1, y0:y1, z0:z1] = color
    data[x0:x1, y0, z0:z1] = color
    data[x0:x1, y1, z0:z1] = color
    data[x0:x1, y0:y1, z0] = color
    data[x0:x1, y0:y1, z1] = color
    return data

def work_nrrd(fpath):
    data, opt = nrrd.read(fpath)
    print data.shape
    print opt
    view_3d_data_simple(data)

def main():
    parser = argparse.ArgumentParser(
        description='demo for reading images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('img_fpath')
    parser.add_argument('--nrrd', action='store_true',
                        help='read nrrd files')
    args = parser.parse_args()

    if args.nrrd:
        return work_nrrd(args.img_fpath)

    reader = ScenePackReader(args.img_fpath)
    img = reader.scenes[0].objects['zhang gui feng-before-vein']
    surface = reader.scenes[0].objects['tumor']
    view_3d_data_simple(draw_surface_box_on_image(surface, img))

if __name__ == '__main__':
    main()
