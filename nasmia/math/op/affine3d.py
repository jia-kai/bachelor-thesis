# -*- coding: utf-8 -*-
# $File: affine3d.py
# $Date: Fri May 01 13:24:01 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
import numpy as np

import os

NR_THREAD_PER_BLOCK = 128

DEFAULT_NVCC_FLAGS.append('-ccbin=g++-4.8')
CACHE_DIR = '/tmp/nasmia_pycuda_cache'
if not os.path.isdir(CACHE_DIR):
    os.mkdir(CACHE_DIR)

mod = SourceModule(
    open(os.path.join(os.path.dirname(__file__), 'affine3d_kern.cu')).read(),
    no_extern_c=True, cache_dir=CACHE_DIR)
func = mod.get_function('batched_affine3d')
func.set_cache_config(drv.func_cache.PREFER_L1)


def batched_affine3d(src, affine_mat, oshape=None,
                     affine_mat_src_to_dest=False):
    """:param affine_mat_src_to_dest: whether *affine_mat* maps from src coord
        to dest coord, so it should be ivnersed for further processing"""

    fmtarr = lambda v: np.ascontiguousarray(np.asarray(v).astype(np.float32))
    src = fmtarr(src)
    if affine_mat.ndim == 2:
        affine_mat = np.tile(affine_mat, (src.shape[0], 1, 1))
    affine_mat = fmtarr(affine_mat)
    assert (src.ndim == 4 and affine_mat.ndim == 3 and
            affine_mat.shape[0] == src.shape[0] and
            affine_mat.shape[1:] == (4, 4)), \
        'bad shape: src={} affine_mat={}'.format(src.shape, affine_mat.shape)

    if oshape is None:
        oshape = src.shape[1:]
    assert len(oshape) == 3
    dest = np.empty([src.shape[0]] + list(oshape), dtype=np.float32)
    dest.fill(np.nan)

    computing_size = dest.size / np.min(dest.shape[1:])
    block = (min(NR_THREAD_PER_BLOCK, computing_size), 1, 1)
    grid = ((computing_size - 1) / block[0] + 1, 1)

    iargs = list(map(
        np.int32,
        [dest.size] + list(dest.shape[1:]) + list(src.shape[1:])))

    if affine_mat_src_to_dest:
        affine_mat_inv = np.empty_like(affine_mat)
        for i in range(affine_mat.shape[0]):
            affine_mat_inv[i] = np.linalg.inv(affine_mat[i])
        affine_mat = affine_mat_inv

    func(drv.Out(dest), drv.In(src), drv.In(affine_mat), *iargs,
         block=block, grid=grid)

    assert np.isfinite(dest.sum())
    return dest


def make_rot3d_uniform(nr, rng=np.random, center=None):
    """make affine matrices for random 3D rotation by uniformly sampling from
    quaternions"""

    # sample uniformly on sphere, rotation axis
    rv = rng.normal(size=(nr, 3))
    rv /= np.sqrt(np.square(rv).sum(axis=1, keepdims=True))

    # angle
    phi = rng.uniform(low=0, high=np.pi*2, size=(nr, ))

    nx, ny, nz = rv.T
    nx2, ny2, nz2 = map(np.square, [nx, ny, nz])
    s = np.sin(phi)
    c = np.cos(phi)
    opc = 1 - c

    tz = nx * ny * opc
    ty = nx * nz * opc
    tx = ny * nz * opc

    nxs = nx * s
    nys = ny * s
    nzs = nz * s

    m11 = nx2 + (ny2 + nz2) * c
    m22 = ny2 + (nx2 + nz2) * c
    m33 = nz2 + (nx2 + ny2) * c

    m12 = tz - nzs
    m13 = ty + nys

    m21 = tz + nzs
    m23 = tx - nxs

    m31 = ty - nys
    m32 = tx + nxs

    zero = np.zeros_like(m11)
    one = np.ones_like(m11)

    m = [[m11, m12, m13, zero],
         [m21, m22, m23, zero],
         [m31, m32, m33, zero],
         [zero, zero, zero, one]]

    m = np.transpose(m, axes=(2, 0, 1))
    if center:
        x, y, z = map(float, center)
        trans0 = np.eye(4)
        trans1 = np.eye(4)
        trans0[:3, 3] = -x, -y, -z
        trans1[:3, 3] = x, y, z
        for i in m:
            i[:] = trans1.dot(i).dot(trans0)
    return m
