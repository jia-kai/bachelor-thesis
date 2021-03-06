#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: get_roc.py
# $Date: Fri Jun 12 10:50:52 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import pyximport
pyximport.install()

from get_roc_impl import get_roc

import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description='get ROC of a single test image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--border_dir',
                        default=os.path.abspath(os.path.join(
                            os.path.dirname(__file__),
                            '../../sliver07/border-dist')),
                        help='directory that store the border dist files')
    parser.add_argument('--dedup_grid_size', type=int, default=2,
                        help='matched points that are placed inside the same'
                        ' grid would be considered as duplications')
    parser.add_argument('--dist_min', type=int, default=-1,
                        help='minimum dist to be considered as hit')
    parser.add_argument('--dist_max', type=int, default=1,
                        help='maximum dist to be considered as hit')
    parser.add_argument('--nr_point', type=int, default=10000,
                        help='number of points on ROC curve')
    parser.add_argument('--plot', action='store_true',
                        help='plot ROC curve')
    parser.add_argument('-o', '--output',
                        help='write text ROC curve to output')
    parser.add_argument('--border_dist_stat',
                        help='write stats of feature dist vs border dist')
    parser.add_argument('match_result', nargs='+',
                        help='point match results of training images '
                        'on this test image')
    parser.add_argument('--dump_dist', help='dump calculated dist to file')
    args = parser.parse_args()

    assert args.output or args.plot or args.border_dist_stat,  \
        'please set a least one of --plot, --output, --border_dist_stat'

    roc = get_roc(args)
    if args.output:
        with open(args.output, 'w') as fout:
            for entry in roc:
                fout.write('{}\n'.format(' '.join(map(str, entry))))
    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(roc[:, 0], roc[:, 1])
        plt.show()


if __name__ == '__main__':
    main()
