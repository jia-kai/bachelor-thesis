# -*- coding: utf-8 -*-
# $File: conv_splitter.py
# $Date: Sun Apr 05 15:57:25 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from . import ProgressReporter

import numpy as np

import logging
logger = logging.getLogger(__name__)

class ConvSplitter(object):
    """split np.ndarray for convolution, and merge result"""
    _axis_start = None
    _axis_end = None
    _split = None
    _kern_shape = None
    _callback = None
    _out_chl = None
    _prog_report = None

    verbose = False
    progress = False

    def __init__(self, axis_start, axis_end, split, kern_shape, out_chl,
                 callback):
        """:param split: int, or list of length axis_end - axis_start
        :param callback: function to be called on each part"""
        if not isinstance(split, list):
            split = (int(split), ) * (axis_end - axis_start)
        split = list(map(int, split))
        assert len(split) == axis_end - axis_start
        self._axis_start = int(axis_start)
        self._axis_end = int(axis_end)
        assert self._axis_start > 0
        self._split = split
        self._kern_shape = kern_shape
        self._out_chl = int(out_chl)
        self._callback = callback

    def __call__(self, x):
        assert x.ndim >= self._axis_end

        oshape = list(x.shape)
        oshape[self._axis_start - 1] = self._out_chl
        for i in range(self._axis_start, self._axis_end):
            oshape[i] = (
                x.shape[i] - self._kern_shape[i - self._axis_start] + 1)
        output = np.empty(oshape, dtype=x.dtype)
        output.fill(np.nan)

        if self.progress:
            self._prog_report = ProgressReporter('split', np.prod(self._split))
        try:
            self._work(x, output, self._axis_start)
        finally:
            if self.progress:
                self._prog_report.finish()
        assert not np.isnan(output.sum())
        return output

    def _work(self, x, y, axis):
        if axis == self._axis_end:
            if self.verbose:
                logger.info('work on subarray of shape: {}'.format(x.shape))
            if self.progress:
                self._prog_report.trigger()
            y[:] = self._callback(x)
            return

        axis_rela = axis - self._axis_start
        kern_shape = self._kern_shape[axis_rela]
        out_size = x.shape[axis] - kern_shape + 1
        assert out_size == y.shape[axis], (out_size, y.shape, axis)
        out_part_size = (out_size - 1) / self._split[axis_rela] + 1
        np_idx = [slice(None)] * x.ndim
        for i in range(0, out_size, out_part_size):
            it = min(i + out_part_size, out_size)
            np_idx[axis] = slice(i, it)
            ysub = y[np_idx]
            np_idx[axis] = slice(i, it + kern_shape - 1)
            xsub = x[np_idx]
            self._work(xsub, ysub, axis + 1)
