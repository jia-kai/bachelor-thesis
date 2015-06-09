#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: get_auc.py
# $Date: Tue Jun 09 13:32:41 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import nasmia

from scipy.integrate import cumtrapz
import numpy as np

import os
import argparse
import logging
logger = logging.getLogger(__name__)

MIN_X = 5e-2

def get_auc(x, y, avg=False):
    x, y = map(np.asarray, (x, y))
    mask = x >= MIN_X
    x = x[mask]
    y = y[mask]
    rst = float(cumtrapz(y, x)[-1])
    if avg:
        rst /= x.max() - x.min()
    return rst

def get_auc_load(fpath):
    x = []
    y = []
    with open(fpath) as fin:
        for line in fin:
            line = map(float, line.split())
            x.append(line[0])
            y.append(line[1])
    x, y = map(np.array, (x, y))
    return get_auc(x, y)

def main():
    parser = argparse.ArgumentParser(
        description='calculate AUC for ROCs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()

    for i in args.input:
        logger.info('{}: {}'.format(os.path.basename(i), get_auc_load(i)))

if __name__ == '__main__':
    main()
