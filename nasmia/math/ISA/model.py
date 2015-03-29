# -*- coding: utf-8 -*-
# $File: model.py
# $Date: Sun Mar 29 09:40:20 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np

class ISAModel(object):
    bias = None
    coeff = None
    outhid_conn = None

    def __init__(self, bias, coeff, outhid_conn):
        self.bias = bias.reshape(-1, 1)
        self.coeff = coeff
        self.outhid_conn = outhid_conn

    def __call__(self, data, level2=True):
        chg_shape = False
        if data.ndim == 1:
            chg_shape = True
            data = data.reshape(-1, 1)
        assert data.ndim == 2 and data.shape[0] == self.bias.size

        hidv = np.square(self.coeff.dot(data + self.bias))
        if level2:
            result = np.sqrt(self.outhid_conn.dot(hidv))
        else:
            result = hidv

        if chg_shape:
            assert result.shape[1] ==  0
            result = result.flatten()
        return result
