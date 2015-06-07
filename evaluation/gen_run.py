#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: gen_run.py
# $Date: Wed Jun 03 18:40:59 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import nasmia

import argparse
import random
import itertools
import logging
import os
logger = logging.getLogger(__name__)

used_names = set()
def append_single(output, desc):
    desc = [i.strip() for i in desc.split('|')]
    if len(desc) == 2:
        desc.append('cos l2')
    assert len(desc) == 3, desc

    cmd = "'{}'".format(desc[0])
    name = "'{}'".format(desc[1])
    metrics = desc[2].strip().split()

    assert name not in used_names
    used_names.add(name)

    output.append('if [ {} ]'.format(' -o '.join([
        '! -e data/pairwise-match/{}-{}'.format(name, i)
        for i in metrics])))
    output.append('then')
    output.append('./step0_extract_feature.sh {} {} -c'.format(cmd, name))
    for i in metrics:
        output.append('./step1_pairwise_match.sh {} {} -c'.format(name, i))
    output.append('fi')
    output.append('rm -f data/feature/{}/*.* || true'.format(name))

    for i in metrics:
        output.append('./step2_get_roc.sh {}-{}'.format(name, i))

    for i in metrics:
        if not os.path.exists('data/roc/{}-{}.txt'.format(desc[1], i)):
            return True
    logger.info('ROC for {} already exists'.format(desc[1:]))

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

    random.Random(20150531).shuffle(lines)

    gpus = args.gpus.split(',')
    output = [[
        '#!/bin/bash -e',
        'export THEANO_FLAGS=device=gpu{}'.format(i)] for i in gpus]

    oiter = itertools.cycle(output)

    cur_out = next(oiter)
    for line in lines:
        if append_single(cur_out, line):
            cur_out = next(oiter)

    for idx, val in enumerate(output):
        fpath = 'data/run{}.sh'.format(idx)
        val.append('')
        with open(fpath, 'w') as fout:
            fout.write('\n'.join(val))
        os.chmod(fpath, 0755)

if __name__ == '__main__':
    main()
