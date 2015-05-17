#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: plot_roc.py
# $Date: Sun May 17 18:27:27 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import numpy as np

from itertools import cycle
import argparse
import sys
import os.path

EPS = 1e-8

def get_marker_cycler():
    if False:
        markers = []
        for m in Line2D.markers:
            try:
                if len(m) == 1 and m not in [' ', '|', '_', ',', '+', 'x', '.',
                                             '1', '2', '3', '4', '*']:
                    markers.append(m)
            except TypeError:
                pass
        print markers
    markers = ['D', 's', '^', 'd', 'o', 'p', 'v', '<', '>', 'h']
    return cycle(markers)

def read_data(args, fpath):
    xdata = []
    ydata = []
    with open(fpath) as fin:
        for line in fin:
            if line.startswith('#'):
                continue
            line = map(float, line.split())
            if line[0] >= args.xmin:
                xdata.append(line[0])
                ydata.append(line[1])

    return xdata, ydata

def label_from_filename(fname):
    s = os.path.basename(fname)
    if '.' in s:
        s = s[:s.find('.')]
    s = s.replace('-l2', '-$L_2$')
    return s

def main():
    parser = argparse.ArgumentParser(
        description='plot ROC data files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loc', default='best',
                        help='location of legend')
    parser.add_argument('--trleg', action='store_true',
                        help='transparent legend box')
    parser.add_argument('--xlabel', help='X axis label',
                        default='top ratio')
    parser.add_argument('--ylabel', help='Y axis label',
                        default='precision')
    parser.add_argument('--xlog', action='store_true',
                        help='use log scale for X axis')
    parser.add_argument('--xmin', default=1e-3, type=float,
                        help='minimum value for x')
    parser.add_argument('-o', '--output', metavar='output', help='output file')
    parser.add_argument('data_fpath', nargs='+')
    args = parser.parse_args()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    #ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, which='major')

    if args.xlog:
        ax.set_xscale('log')

    markers = get_marker_cycler()

    data_fpath = [(i, label_from_filename(i)) for i in args.data_fpath]

    for fname, label in data_fpath:
        x, y = read_data(args, fname)
        marker = next(markers)
        ax.plot(x, y, marker=marker, label=label, linewidth=2,
                markevery=len(x)/5, markersize=10, markeredgecolor='none')

    leg = plt.legend(loc=args.loc, fancybox=True, numpoints=1)
    if args.trleg:
        leg.get_frame().set_alpha(0.5)

    if args.output:
        fig.savefig(args.output)

    plt.show()

if __name__ == '__main__':
    main()
