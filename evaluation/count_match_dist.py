#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: count_match_dist.py
# $Date: Sun May 31 16:37:14 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.io import PointMatchResult

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='count number of points w.r.t. different dist',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--border', required=True,
                        help='border dist file corresponding to match result')
    parser.add_argument('-m', '--match_result', required=True,
                        help='point match result')
    parser.add_argument('-o', '--output', help='write result to output')
    parser.add_argument('--plot', action='store_true', help='plot result')
    args = parser.parse_args()

    mtch_rst = serial.load(args.match_result, PointMatchResult)
    border_dist = serial.load(args.border, np.ndarray)
    assert mtch_rst.img_shape == border_dist.shape, (
        mtch_rst.img_shape, border_dist.shape)

    stat = defaultdict(int)
    for k in range(mtch_rst.dist.shape[0]):
        x, y, z = mtch_rst.idx[k]
        stat[border_dist[x, y, z]] += 1

    stat = np.array(sorted(stat.iteritems()), dtype=np.float32)
    stat[:, 0] -= mtch_rst.args.ref_dist
    stat[:, 1] *= 1.0 / mtch_rst.dist.size

    if args.plot:
        plt.plot(stat[:, 0], stat[:, 1], 'x-')
        plt.show()

    if args.output:
        with open(args.output, 'w') as fout:
            for x, y in stat:
                fout.write('{} {}\n'.format(x, y))

if __name__ == '__main__':
    main()
