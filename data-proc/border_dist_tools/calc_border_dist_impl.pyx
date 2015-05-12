# -*- coding: utf-8 -*-
# $File: calc_border_dist_impl.pyx
# $Date: Tue May 12 16:12:24 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np
cimport numpy as np
cimport cython

import itertools

ctypedef np.int32_t DTYPE_t

cdef struct Coord:
    int x, y, z


# @cython.boundscheck(False) #only a little faster
def calc_border_dist(np.ndarray[DTYPE_t, ndim=3] mask):
    cdef np.ndarray[DTYPE_t, ndim=3] result = np.empty_like(mask)
    result.fill(np.iinfo(np.int32).max)

    # convert npy_intp* to tuple
    mask_shape = (mask.shape[0], mask.shape[1], mask.shape[2])

    cdef unsigned qh = 0, qt = 0, qlen = mask.size
    cdef np.ndarray[np.uint8_t] queue_mem = np.empty(
        (qlen * sizeof(Coord), ), dtype='uint8')
    cdef np.ndarray[np.int32_t, ndim=3] in_queue = np.zeros(
        mask_shape, dtype=np.int32)
    cdef Coord* queue = <Coord*>(queue_mem.data)

    cdef unsigned i, j, k
    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            for k in range(1, mask.shape[2] - 1):
                if (mask[i, j, k] and not (
                        mask[i - 1, j - 1, k - 1] and mask[i - 1, j - 1, k]
                        and mask[i - 1, j - 1, k + 1] and mask[i - 1, j, k - 1]
                        and mask[i - 1, j, k] and mask[i - 1, j, k + 1]
                        and mask[i - 1, j + 1, k - 1] and mask[i - 1, j + 1, k]
                        and mask[i - 1, j + 1, k + 1] and mask[i, j - 1, k - 1]
                        and mask[i, j - 1, k] and mask[i, j - 1, k + 1]
                        and mask[i, j, k - 1] and mask[i, j, k + 1]
                        and mask[i, j + 1, k - 1] and mask[i, j + 1, k]
                        and mask[i, j + 1, k + 1] and mask[i + 1, j - 1, k - 1]
                        and mask[i + 1, j - 1, k] and mask[i + 1, j - 1, k + 1]
                        and mask[i + 1, j, k - 1] and mask[i + 1, j, k]
                        and mask[i + 1, j, k + 1] and mask[i + 1, j + 1, k - 1]
                        and mask[i + 1, j + 1, k] and mask[i + 1, j + 1, k + 1]
                )):
                    queue[qt].x = i
                    queue[qt].y = j
                    queue[qt].z = k
                    result[i, j, k] = 0
                    in_queue[i, j, k] = 1
                    qt += 1

    cdef int di, dj, dk, dist1
    cdef unsigned i1, j1, k1
    while qh != qt:
        i = queue[qh].x
        j = queue[qh].y
        k = queue[qh].z
        in_queue[i, j, k] = 0
        qh += 1
        if qh == qlen:
            qh = 0

        dist1 = result[i, j, k] + 1
        for di, dj, dk in itertools.product([-1, 0, 1], repeat=3):
            i1 = i + di
            j1 = j + dj
            k1 = k + dk
            if (i1 < mask.shape[0]
                    and j1 < mask.shape[1]
                    and k1 < mask.shape[2]
                    and dist1 < result[i1, j1, k1]):
                result[i1, j1, k1] = dist1
                if not in_queue[i1, j1, k1]:
                    queue[qt].x = i1
                    queue[qt].y = j1
                    queue[qt].z = k1
                    in_queue[i1, j1, k1] = 1
                    qt += 1
                    if qt == qlen:
                        qt = 0
                    assert qt != qh

    cdef int max_r = max(mask_shape)
    for i, j, k in itertools.product(*map(range, mask_shape)):
        r = result[i, j, k]
        assert r >= 0 and r <= max_r
        if not mask[i, j, k]:
            result[i, j, k] = -r
    return result
