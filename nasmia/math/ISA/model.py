# -*- coding: utf-8 -*-
# $File: model.py
# $Date: Wed Apr 01 22:17:08 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from ..op import sharedX

import numpy as np

class ISAModel(object):
    coeff = None
    """coeff to be multiplied to input"""

    bias = None
    """bias to be added to each channel after input multiplied by coeff"""

    outhid_conn = None

    def __init__(self, coeff, bias, outhid_conn):
        self.coeff = coeff
        self.bias = bias
        self.outhid_conn = outhid_conn
        assert bias.ndim == 1 and coeff.ndim == 2
        assert coeff.shape[0] == bias.size

    def __call__(self, data, level2=True):
        chg_shape = False
        if data.ndim == 1:
            chg_shape = True
            data = data.reshape(-1, 1)
        assert data.ndim == 2 and data.shape[0] == self.coeff.shape[1]

        hidv = np.square(self.coeff.dot(data) + self.bias.reshape(-1, 1))
        if level2:
            result = np.sqrt(self.outhid_conn.dot(hidv))
        else:
            result = hidv

        if chg_shape:
            assert result.shape[1] ==  0
            result = result.flatten()
        return result

    def get_conv_coeff(self, kern_shape=None, in_chl=1):
        """convert coeff to theano theano.sandbox.cuda.blas.GpuCorr3dMM format;
        more specifically, in (out channel, in channel, x, y, z) format, without
        flipping
        assume image format: batch, chl, x, y, z
        :param kern_shape: kernel shape, if None, would be the cubic root of
            kernel size"""
        if kern_shape is None:
            kern_size = (self.coeff.shape[1] / in_chl) ** (1.0 / 3)
            kern_shape = [int(kern_size + 0.5)] * 3
        assert len(kern_shape) == 3
        actual_dim = np.product(kern_shape) * in_chl
        assert actual_dim == self.coeff.shape[1], \
            'bad shape: {} {}'.format(self.coeff.shape, actual_dim)
        return self.coeff.reshape([self.coeff.shape[0], in_chl] + kern_shape)

    def fprop_conv(self, state_below):
        """fprop theano state_below using conv method
        :param state_below: batch, chl, x, y, z"""
        import theano
        import theano.tensor as T
        from theano.sandbox.cuda.blas import GpuCorr3dMM
        assert state_below.ndim == 5
        W = self.get_conv_coeff()
        corr = GpuCorr3dMM()
        conv_rst = corr(state_below, sharedX(self.get_conv_coeff()))
        conv_rst += sharedX(self.bias).dimshuffle('x', 0, 'x', 'x', 'x')
        sqr = T.square(conv_rst)
        vsum = T.tensordot(sharedX(self.outhid_conn), sqr, axes=[1, 1])
        vsum = vsum.dimshuffle(1, 0, 2, 3, 4)
        return T.sqrt(vsum)
