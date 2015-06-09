#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: gen_stat.py
# $Date: Tue Jun 09 15:08:27 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from get_auc import get_auc, MIN_X

import numpy as np

import argparse
import os
from copy import deepcopy
from collections import namedtuple

STAT_ENTRY_KEYS = ['auc', 'mauc', 'max_top_ratio', 'min_precision',
                   'max_precision']
StatEntry = namedtuple('StatEntry', ['name'] + STAT_ENTRY_KEYS)

def reformat_col(data, name, use_max=False, top=3, fmt='${:.3f}$'.format):
    col_data = []

    for idx, i in enumerate(data):
        v = getattr(i, name)
        if use_max:
            v = -v
        col_data.append((v, i, idx))

    col_data.sort()

    for rank, (_, entry, orig_idx) in enumerate(col_data):
        v = fmt(getattr(entry, name))
        v = '{} ({})'.format(v, rank + 1)
        if rank < top:
            v = r'{{\bf {}}}'.format(v)
        data[orig_idx] = entry._replace(**{name: v})


def write_output(fout, data):
    data = deepcopy(data)
    for key in STAT_ENTRY_KEYS:
        reformat_col(data, key, use_max=True)

    for i in data:
        fout.write('&'.join(i))
        fout.write('\\\\\n')


def main():
    parser = argparse.ArgumentParser(
        description='generate stat table in latex format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()

    all_data = []
    for i in args.input:
        with open(i) as fin:
            x = []
            y = []
            for line in fin:
                line = map(float, line.split())
                if line[0] >= MIN_X:
                    x.append(line[0])
                    y.append(line[1])
            x, y = map(np.array, (x, y))

        name = os.path.basename(i)
        name = name[:name.rfind('.')]
        all_data.append(StatEntry(
            name, get_auc(x, y), get_auc(x, y, True),
            x.max(), y.min(), y.max()))

    with open(args.output, 'w') as fout:
        write_output(fout, all_data)

if __name__ == '__main__':
    main()
