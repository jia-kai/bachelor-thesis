# -*- coding: utf-8 -*-
# $File: calc_border_dist_impl.pyx
# $Date: Mon May 11 19:42:50 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np
cimport numpy as np

import itertools

ctypedef np.int_t DTYPE_t

cdef struct Coord:
    int x, y, z


def calc_border_dist(np.ndarray[DTYPE_t, ndim=3] mask):
    cdef np.ndarray[DTYPE_t, ndim=3] result = np.empty_like(mask)
    result.fill(-1)

    cdef np.ndarray[np.uint8_t] queue_mem = np.empty(
        (mask.size * sizeof(Coord), ), dtype='uint8')
    cdef Coord* queue = <Coord*>(queue_mem.data)
    cdef unsigned qh = 0, qt = 0

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
                    qt += 1

    cdef int di, dj, dk, dist1, dist_orig
    cdef unsigned i1, j1, k1
    while qh < qt:
        i = queue[qh].x
        j = queue[qh].y
        k = queue[qh].z
        qh += 1
        dist1 = result[i, j, k] + 1
        for di, dj, dk in itertools.product([-1, 0, 1], repeat=3):
            i1 = i + di
            j1 = j + dj
            k1 = k + dk
            if (i1 >= 1 and i1 < mask.shape[0] - 1
                    and j1 >= 1 and j1 < mask.shape[1] - 1
                    and k1 >= 1 and k1 < mask.shape[2] - 1 and
                    mask[i1, j1, k1]):
                dist_orig = result[i1, j1, k1]
                if dist_orig == -1 or dist1 < dist_orig:
                    result[i1, j1, k1] = dist1
                    queue[qt].x = i1
                    queue[qt].y = j1
                    queue[qt].z = k1
                    qt += 1

    return result
