# -*- coding: utf-8 -*-
# $File: patch_cropper.py
# $Date: Sun May 10 20:59:54 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np

class CropPatchHelper(object):
    _rng = None
    _patch_size = None
    _bg_extend = None
    _used = None
    _mask = None
    _cur_dynrange_thresh = None

    intensity_thresh_coeff = 0.6
    dynrange_thresh_coeff = 0.02
    max_overlap = 0.5
    zero_thresh = 50
    mask_max_zero_ratio = 0.1
    max_zero_ratio = 0.36

    def __init__(self, patch_size, bg_extend, rng):
        """:param bg_extend: border width for background patch"""
        self._rng = rng
        self._patch_size = int(patch_size)
        self._bg_extend = int(bg_extend)

    def __call__(self, data, mask=None):
        """:return: generator to iterate through subpatches"""
        offset = -data.min()
        if offset:
            data = data + offset

        if mask is None:
            axrange = self._find_axis_range(
                data, data.mean() * self.intensity_thresh_coeff)
        else:
            assert mask.shape == data.shape
            mask = (mask != 0).astype(np.int32)
            axrange = self._find_axis_range(mask, 0.01)
        self._mask = mask

        self._cur_dynrange_thresh = data.max() * self.dynrange_thresh_coeff
        for idx, (i, j) in enumerate(axrange):
            assert j - i > self._patch_size * 2, (idx, i, j)
            axrange[idx] = (i, j - self._patch_size + 1)

        self._used = np.zeros_like(data, dtype='float32')

        def gen_subidx():
            r = lambda v: self._rng.randint(*v)
            return tuple([slice(v, v + self._patch_size)
                   for v in map(r, axrange)])

        while True:
            while True:
                idx = gen_subidx()
                sub = data[idx]
                if self._check_patch(sub, idx):
                    break
            self._used[idx] += 1.0 / sub.size
            if self._bg_extend:
                sub = data[self._extend_np_idx(idx)]
            if offset:
                sub = sub - offset
            yield sub

    def _extend_np_idx(self, idx):
        rst = []
        for i in idx:
            assert isinstance(i, slice)
            s1 = i.start - self._bg_extend
            t1 = i.stop + self._bg_extend
            assert s1 >= 0
            rst.append(slice(s1, t1))
        return tuple(rst)

    def _check_patch(self, patch, patch_idx):
        if self._used[patch_idx].mean() > self.max_overlap:
            return False
        if (self._mask is not None and
                1 - self._mask[patch_idx].mean() >= self.mask_max_zero_ratio):
            return False
        if (patch <= self.zero_thresh).mean() >= self.max_zero_ratio:
            return False
        if patch.max() - patch.min() < self._cur_dynrange_thresh:
            return False
        return True

    def _find_axis_range(self, data, mean_thresh):
        rst = []
        test = lambda v: v.mean() > mean_thresh

        for axis in range(data.ndim):
            def visit(v):
                idx = [slice(None)] * data.ndim
                idx[axis] = v
                tup = tuple(idx)
                return data[tup]
            low = self._bg_extend
            while not test(visit(low)):
                low += 1
            high = data.shape[axis] - 1 - self._bg_extend
            while not test(visit(high)):
                high -= 1
            assert low < high
            rst.append((low, high + 1))
        return rst
