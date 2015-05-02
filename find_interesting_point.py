#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: find_interesting_point.py
# $Date: Tue Apr 21 23:03:42 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.visualize import view_3d_data_simple, draw_box_on_image
from nasmia.math.ISA.config import LAYER1_PATCH_SIZE

import numpy as np
import matplotlib.pyplot as plt

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='find interesting feature points',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('pack')
    parser.add_argument('-t', '--thresh', type=float)
    args = parser.parse_args()

    pack = serial.load(args.pack)
    mag = pack['ftr'].mean(axis=0)
    print 'ftr shape:', mag.shape

    img = pack['img']
    if args.thresh:
        thresh = sorted(mag.flatten())[int(mag.size * (1 - args.thresh))]
        d = LAYER1_PATCH_SIZE / 2
        color = (0, img.max(), 0)
        for x, y, z in np.transpose((mag >= thresh).nonzero()):
            x, y, z = x + d, y + d, z + d
            img = draw_box_on_image(
                img, (x - 2, y - 2, z - 2), (x + 2, y + 2, z + 2), color)
    view_3d_data_simple(img)


    fig = plt.figure()
    plt.hist(mag.flatten(), 50)
    plt.show()

if __name__ == '__main__':
    main()
