#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: plot_roc.py
# $Date: Fri Jun 12 10:40:30 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from get_auc import MIN_X, get_auc

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

def get_color_cycler():
    # http://tableaufriction.blogspot.ro/2012/11/finally-you-can-use-tableau-data-colors.html
    tableau10 = np.array(
        [(255, 188, 121), (95, 158, 209), (255, 128, 14),
         (0, 107, 164), (207, 207, 207), (200, 82, 0),
         (171, 171, 171), (162, 200, 236), (137, 137, 137), (89, 89, 89)],
        dtype=np.float32)

    tableau10 *= 0.9 / 255.0

    return cycle(tableau10)

def read_data(args, fpath):
    xdata = []
    ydata = []
    xerr = []
    yerr = []
    with open(fpath) as fin:
        for line in fin:
            if line.startswith('#'):
                continue
            line = map(float, line.split())
            if line[0] < MIN_X:
                continue
            xdata.append(line[0])
            ydata.append(line[1])
            xerr.append(line[3])
            yerr.append(line[4])

    return map(np.array, (xdata, ydata, xerr, yerr))

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
    parser.add_argument('--no_errbar', action='store_true',
                        help='do not plot error bar')
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
    colors = get_color_cycler()

    data_fpath = [(i, label_from_filename(i)) for i in args.data_fpath]

    all_data = []
    for fname, label in data_fpath:
        x, y, xerr, yerr = read_data(args, fname)
        auc = get_auc(x, y, avg=True)
        label = '{}({:.3f})'.format(label, auc)
        all_data.append((-auc, x, y, yerr, label))

    all_data.sort()

    for _, x, y, yerr, label in all_data:
        marker = next(markers)
        cur_color = next(colors)
        ax.plot(x, y, marker=marker, label=label, linewidth=2,
                markevery=0.3, markersize=10,
                markeredgecolor='none', color=cur_color)
        if not args.no_errbar:
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15,
                            edgecolor='none', facecolor=cur_color)


    leg = plt.legend(loc=args.loc, fancybox=True, numpoints=1)
    if args.trleg:
        leg.get_frame().set_alpha(0.5)

    if args.output:
        fig.savefig(args.output, bbox_inches='tight')
    else:
        plt.show()

if __name__ == '__main__':
    main()
