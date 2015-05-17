# -*- coding: utf-8 -*-
# $File: get_roc_impl.pyx
# $Date: Sun May 17 16:52:48 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.utils import serial
from nasmia.io import KNNResult

import numpy as np
cimport numpy as np

import os
import logging
logger = logging.getLogger(__name__)

cdef load_file_list(all_data, flist, test_img_name, args):
    """:return: list of (is_tp, thresh)"""

    border_dist_file = os.path.join(args.border_dir, test_img_name + '.nii.gz')
    logger.info('use border dist: {}'.format(border_dist_file))
    cdef np.ndarray[np.int32_t, ndim=3] border_dist = serial.load(
        border_dist_file, np.ndarray)

    cdef np.ndarray[np.int32_t, ndim=3] knn_idx
    cdef np.ndarray[np.float32_t, ndim=2] knn_dist
    cdef unsigned ptnum, nnnum, nnnum_max, x, y, z
    cdef int ref_dist, geo_dist, min_geo_dist, max_geo_dist, is_tp
    min_geo_dist = args.dist_min
    max_geo_dist = args.dist_max
    for i in flist:
        logger.info('load {}'.format(i))
        knn = serial.load(i, KNNResult)
        knn_idx = knn.idx
        knn_dist = knn.dist
        nnnum_max = min(knn_dist.shape[1], args.select_knn)
        ref_dist = knn.args.ref_dist
        for ptnum in range(knn_dist.shape[0]):
            for nnnum in range(nnnum_max):
                x, y, z = knn_idx[ptnum, nnnum]
                geo_dist = border_dist[x, y, z] - ref_dist
                is_tp = geo_dist >= min_geo_dist and geo_dist <= max_geo_dist
                all_data.append((is_tp, knn_dist[ptnum, nnnum]))


cdef load_all_data(args):
    all_data = []
    flist = sorted([(os.path.basename(i).split('.')[0].split('-'), i)
                    for i in args.knn_result])

    for key in sorted(set(i[0][1] for i in flist)):
        sub_flist = [i[1] for i in flist if i[0][1] == key]
        load_file_list(all_data, sub_flist, key, args)

    return all_data

def get_roc(args):
    """:return: np array (nr, 3), (point ratio, precision, dist thresh)"""

    cdef np.ndarray[np.float32_t, ndim=2] all_data = np.array(
        load_all_data(args), dtype='float32')
    assert all_data.ndim == 2
    all_data = all_data[all_data[:, 1].argsort()]

    cdef np.ndarray[np.float32_t, ndim=1] uniq_dist = np.unique(
        all_data[:, 1])

    if uniq_dist.size >= args.nr_point * 2:
        uniq_dist = uniq_dist[::uniq_dist.size / args.nr_point]

    logger.info('calc ROC; number of points: {}'.format(uniq_dist.size))

    cdef np.ndarray[np.float32_t, ndim=2] result = np.empty(
        (uniq_dist.size, 3), dtype='float32')
    cdef float thresh, tot_nr_tp = all_data[:, 0].sum(), nr_tp = 0, idx_src_f
    cdef unsigned idx, idx_src = 0

    for idx in range(uniq_dist.size):
        thresh = uniq_dist[idx]
        while idx_src < all_data.shape[0] and all_data[idx_src, 1] <= thresh:
            nr_tp += all_data[idx_src, 0]
            idx_src += 1

        idx_src_f = idx_src
        result[idx] = (idx_src_f / all_data.shape[0], nr_tp / idx_src_f,
                       thresh)

    return result
