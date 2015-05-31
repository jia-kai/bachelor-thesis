#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: gen_run.py
# $Date: Sun May 31 20:09:22 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import argparse
import random
import itertools
import os

def append_single(output, desc):
    desc = desc.split('|')
    if len(desc) == 2:
        desc.append('cos')
    assert len(desc) == 3, desc

    cmd = "'{}'".format(desc[0].strip())
    name = "'{}'".format(desc[1].strip())
    metrics = desc[2].strip().split()

    output.append('./step0_extract_feature.sh {} {}'.format(cmd, name))
    for i in metrics:
        output.append('./step1_extract_feature.sh {} {}'.format(name, i))
    output.append('rm -rf data/feature/{}'.format(name))

    for i in metrics:
        output.append('./step2_get_roc.sh {}-{}'.format(name, i))

def main():
    parser = argparse.ArgumentParser(
        description='generate model runner scripts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-G', '--gpus', required=True,
                        help='list of gpus to use')
    args = parser.parse_args()

    with open('list') as fin:
        lines = [i.strip() for i in fin.readlines()]
        lines = [i for i in lines if i]

    random.shuffle(lines)

    gpus = args.gpus.split(',')
    output = [[
        '#!/bin/bash -e',
        'export THEANO_FLAGS=device=gpu{}'.format(i)] for i in gpus]

    oiter = itertools.cycle(output)

    for line in lines:
        append_single(next(oiter), line)

    for idx, val in enumerate(output):
        fpath = 'data/run{}.sh'.format(idx)
        val.append('')
        with open(fpath, 'w') as fout:
            fout.write('\n'.join(val))
        os.chmod(fpath, 0755)

if __name__ == '__main__':
    main()
