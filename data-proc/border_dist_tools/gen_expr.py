#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: gen_expr.py
# $Date: Mon May 11 19:40:57 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import itertools

def fmt(v):
    if v < 0:
        return ' - {}'.format(-v)
    if not v:
        return ''
    return ' + {}'.format(v)

item = []
for i, j, k in itertools.product([-1, 0, 1], repeat=3):
    if i or j or k:
        i, j, k = map(fmt, (i, j, k))
        item.append('mask[i{}, j{}, k{}]'.format(i, j, k))
for idx, v in enumerate(item):
    if idx % 2 == 1:
        item[idx] = v + '\n'
print (' and '.join(item))
