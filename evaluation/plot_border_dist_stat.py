#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: plot_border_dist_stat.py
# $Date: Fri Jun 12 11:49:04 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from plot_roc import get_marker_cycler, get_color_cycler, label_from_filename

import matplotlib.pyplot as plt

import numpy as np

import argparse
import sys
import os.path

def read_data(fpath):
    x = []
    y = []
    with open(fpath) as fin:
        for line in fin:
            if line.startswith('#'):
                continue
            line = map(float, line.split())
            x.append(line[0])
            y.append(line[1])
    x, y = map(np.array, (x, y))
    y = (y - y.min()) / (y.max() - y.min())
    return x, y

def main():
    parser = argparse.ArgumentParser(
        description='plot raw data files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loc', default='best',
                        help='location of legend')
    parser.add_argument('--trleg', action='store_true',
                        help='transparent legend box')
    parser.add_argument('--xlabel', help='X axis label',
                        default='border distance')
    parser.add_argument('--ylabel', help='Y axis label',
                        default='normalized mean feature distance')
    parser.add_argument('--xmin', help='minimal X value', type=float,
                        default=-40)
    parser.add_argument('--xmax', help='maximal X value', type=float,
                        default=20)
    parser.add_argument('-o', '--output', metavar='output', help='output file')
    parser.add_argument('data_fpath', nargs='+')
    args = parser.parse_args()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.grid(True, which='major')
    ax.set_xlim((args.xmin, args.xmax))
    ax.set_ylim((-0.05, 1.05))

    markers = get_marker_cycler()
    colors = get_color_cycler()

    for fpath in args.data_fpath:
        x, y = read_data(fpath)
        marker = next(markers)
        cur_color = next(colors)
        label = label_from_filename(fpath)
        ax.plot(x, y, marker=marker, label=label, linewidth=2,
                markevery=0.3, markersize=10,
                markeredgecolor='none', color=cur_color)
    ax.axvspan(1, 3, alpha=0.2, facecolor='red', edgecolor='none')

    leg = plt.legend(loc=args.loc, fancybox=True, numpoints=1)
    if args.trleg:
        leg.get_frame().set_alpha(0.5)

    if args.output:
        fig.savefig(args.output, bbox_inches='tight')
    else:
        plt.show()

if __name__ == '__main__':
    main()
