#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: count_dist.py
# $Date: Tue May 12 16:35:47 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.io import KNNResult

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='count number of points w.r.t. different dist',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--border', required=True,
                        help='border dist file corresponding to KNN result')
    parser.add_argument('-k', '--knn_result', required=True,
                        help='KNN result')
    parser.add_argument('-o', '--output', help='write result to output')
    parser.add_argument('--plot', action='store_true', help='plot result')
    args = parser.parse_args()

    knn = serial.load(args.knn_result, KNNResult)
    border_dist = serial.load(args.border, np.ndarray)
    assert knn.img_shape == border_dist.shape, (knn.img_shape,
                                                border_dist.shape)

    stat = defaultdict(int)
    for ptnum in range(knn.dist.shape[0]):
        for k in range(knn.dist.shape[1]):
            x, y, z = knn.idx[ptnum, k]
            stat[border_dist[x, y, z]] += 1

    stat = np.array(sorted(stat.iteritems()), dtype=np.float32)
    stat[:, 0] -= knn.args.ref_dist
    stat[:, 1] *= 1.0 / knn.dist.size

    if args.plot:
        plt.plot(stat[:, 0], stat[:, 1], 'x-')
        plt.show()

    if args.output:
        with open(args.output, 'w') as fout:
            for x, y in stat:
                fout.write('{} {}\n'.format(x, y))

if __name__ == '__main__':
    main()
