# -*- coding: utf-8 -*-
# $File: get_roc_impl.pyx
# $Date: Sun May 31 19:04:49 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.io import PointMatchResult

import numpy as np
cimport numpy as np

import os
import logging
logger = logging.getLogger(__name__)

cdef class ROCDataLoader:
    cdef object _args

    def __init__(self, args):
        self._args = args

    cdef _load_file_list(self, all_data, flist, test_img_name):
        """:return: list of (is_tp, thresh)"""
        args = self._args

        border_dist_file = os.path.join(
            args.border_dir, test_img_name + '.nii.gz')
        logger.info('use border dist: {}'.format(border_dist_file))
        border_dist_np = serial.load(border_dist_file, np.ndarray)
        cdef np.ndarray[np.int32_t, ndim=3] border_dist = border_dist_np

        cdef np.ndarray[np.int32_t, ndim=3] used
        cdef np.ndarray[np.int32_t, ndim=2] match_idx
        cdef np.ndarray[np.float32_t, ndim=1] match_dist

        cdef unsigned ptnum, x, y, z, xg, yg, zg
        cdef unsigned grid_size = args.dedup_grid_size
        cdef int ref_dist, geo_dist, is_tp
        cdef int min_geo_dist = args.dist_min, max_geo_dist = args.dist_max
        for i in flist:
            logger.info('load {}'.format(i))
            match = serial.load(i, PointMatchResult)
            assert match.img_shape == border_dist_np.shape
            match_idx = match.idx.astype(np.int32)
            match_dist = match.dist.astype(np.float32)
            ref_dist = match.args.ref_dist
            used = np.zeros_like(border_dist, dtype=np.int32)
            for ptnum in match_dist.argsort():
                x, y, z = match_idx[ptnum]
                xd, yd, zd = x / grid_size, y / grid_size, z / grid_size
                if used[xd, yd, zd]:
                    continue
                used[xd, yd, zd] = 1
                geo_dist = border_dist[x, y, z] - ref_dist
                is_tp = geo_dist >= min_geo_dist and geo_dist <= max_geo_dist
                all_data.append((is_tp, match_dist[ptnum]))


    def __call__(self):
        """:return: [(is_tp, thresh, idx)]"""
        all_data = []
        flist = sorted([(os.path.basename(i).split('.')[0].split('-'), i)
                        for i in self._args.match_result])

        for key in sorted(set(i[0][1] for i in flist)):
            sub_flist = [i[1] for i in flist if i[0][1] == key]
            self._load_file_list(all_data, sub_flist, key)

        return np.array(all_data, dtype=np.float32)


def get_roc(args):
    """:return: np array (nr, 3), (point num, precision, dist thresh)"""

    data_loader = ROCDataLoader(args)
    cdef np.ndarray[np.float32_t, ndim=2] all_data = data_loader()
    all_data = all_data[all_data[:, 1].argsort()]

    cdef np.ndarray[np.float32_t, ndim=1] uniq_dist = np.unique(
        all_data[:, 1])

    if uniq_dist.size >= args.nr_point * 2:
        uniq_dist = uniq_dist[::uniq_dist.size / args.nr_point]

    logger.info('calc ROC; number of points: {}'.format(uniq_dist.size))

    cdef np.ndarray[np.float32_t, ndim=2] result = np.empty(
        (uniq_dist.size, 3), dtype='float32')
    cdef float thresh, tot_nr_tp = all_data[:, 0].sum(), nr_tp = 0
    cdef unsigned idx, idx_src = 0

    for idx in range(uniq_dist.size):
        thresh = uniq_dist[idx]
        while idx_src < all_data.shape[0] and all_data[idx_src, 1] <= thresh:
            nr_tp += all_data[idx_src, 0]
            idx_src += 1

        result[idx] = (idx_src, nr_tp / idx_src, thresh)

    return result
