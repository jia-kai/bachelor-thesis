#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: test.py
# $Date: Sat Mar 28 15:40:42 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.math.fastISA import FastISA, ISAParam

import numpy as np

nr_data = 500
in_dim = 20
param = ISAParam(in_dim=in_dim, subspace_size=4, out_dim=5)
data = np.random.uniform(size=(nr_data, in_dim))
isa = FastISA(param, data, 2)
wht = isa.get_result()
print np.max(np.abs(wht - np.eye(in_dim)))
