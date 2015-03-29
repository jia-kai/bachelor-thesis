#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: train.py
# $Date: Sun Mar 29 09:51:00 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.math.ISA import ISA, ISAParam
from nasmia.utils import serial

import numpy as np

import argparse
import logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description='train ISA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--subspace_size', type=int, default=2)
    parser.add_argument('--out_dim', type=int, default=300)
    parser.add_argument('--nr_worker', type=int, default=4)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--dump_iter', type=int, default=10,
                        help='number of iters between model dump')
    parser.add_argument('data')
    parser.add_argument('output')
    args = parser.parse_args()

    data = serial.load(args.data, np.ndarray)

    param = ISAParam(
        in_dim=data.shape[0], subspace_size=args.subspace_size,
        out_dim=args.out_dim)
    isa = ISA(param, data, nr_worker=args.nr_worker,
              gpu_list=map(int, args.gpus.split(',')))
    iter_num = 0
    while True:
        monitor = isa.perform_iter(500)
        msg = 'train iter {}\n'.format(iter_num)
        for k, v in monitor.iteritems():
            msg += '{}: {}\n'.format(k, v)
        logger.info(msg[:-1])
        if iter_num % args.dump_iter == 0:
            model = isa.get_model()
            model.monitor = monitor
            serial.dump(model, args.output, use_pickle=True)
        iter_num += 1

if __name__ == '__main__':
    main()
