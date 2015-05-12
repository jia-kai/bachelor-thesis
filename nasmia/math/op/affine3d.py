# -*- coding: utf-8 -*-
# $File: affine3d.py
# $Date: Tue May 12 20:03:58 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from ...utils import serial
import numpy as np

import ctypes
import os
import logging
logger = logging.getLogger(__name__)

_kerndir = os.path.join(os.path.dirname(__file__), 'affine3d_kern')
if os.system('make -C {} >/dev/null'.format(_kerndir)):
    raise RuntimeError('failed to compile')
_lib = ctypes.cdll.LoadLibrary(
    os.path.join(_kerndir, 'libaffine3d.so'))
_kernfunc = _lib.batched_affine3d
_kernfunc.argtypes = (
    [ctypes.POINTER(ctypes.c_float)] * 3 +
    [ctypes.c_int] * 8)
_kernfunc.restype = ctypes.c_int

def _np_ptr(arr):
    assert arr.dtype == np.float32
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def fmtarr(v):
    if (isinstance(v, np.ndarray) and v.flags['C_CONTIGUOUS'] and
            v.dtype == np.float32):
        return v
    return np.ascontiguousarray(np.asarray(v).astype(np.float32))

def batched_affine3d(src, inv_affine_mat, oshape=None, gpuid=0):
    """:param inv_affine_mat: maps coord on dest to src, shape(n, 3, 4)
    :param oshape: tuple of output shape"""

    src = fmtarr(src)
    if inv_affine_mat.ndim == 2:
        inv_affine_mat = np.tile(inv_affine_mat, (src.shape[0], 1, 1))
    inv_affine_mat = fmtarr(inv_affine_mat)
    assert (src.ndim == 4 and inv_affine_mat.ndim == 3 and
            inv_affine_mat.shape[0] == src.shape[0] and
            inv_affine_mat.shape[1:] == (3, 4)), \
        'bad shape: src={} inv_affine_mat={}'.format(
            src.shape, inv_affine_mat.shape)
    assert min(src.shape[1:]) >= 2, 'src image must be at least 2 pixels wide'

    if oshape is None:
        oshape = src.shape[1:]
    assert len(oshape) == 3
    batch = src.shape[0]
    dest = np.empty([batch] + list(oshape), dtype=np.float32)

    ptrargs = list(map(
        _np_ptr,
        [dest, src, inv_affine_mat]))
    iargs = list(map(
        int,
        [batch] + list(dest.shape[1:]) + list(src.shape[1:]) + [gpuid]))

    err = _kernfunc(*(ptrargs + iargs))

    if (not np.isfinite(dest.sum())) or err:
        dump_path = '/tmp/affine3d-bad-param.pkl'
        serial.dump((src, inv_affine_mat, oshape, dest), dump_path)
        raise RuntimeError(
            'batched_affine3d failed: dsum={} err={}, '
            'param dumped to {}'.format(
                dest.sum(), err, dump_path))

    return dest

def afdot(*args):
    """dot two 3x4 affine matrices"""
    assert args
    args = list(args)
    while len(args) > 1:
        m2 = args.pop()
        m1 = args.pop()

        assert m2.shape == m1.shape and m1.shape == (3, 4)

        A1 = m1[:, :3]
        b1 = m1[:, 3:]
        A2 = m2[:, :3]
        b2 = m2[:, 3:]
        rst = np.empty_like(m1)
        rst[:, :3] = A1.dot(A2)
        rst[:, 3] = (A1.dot(b2) + b1).flat
        args.append(rst)
    return args[0]

class RandomAffineMat(object):
    _rng = None
    _nr = None
    _center = None

    min_angle = 0.0
    max_angle = np.pi

    min_scale = 0.9
    max_scale = 1.1

    def __init__(self, nr, center=None, rng=np.random,
                 **kwargs):
        self._rng = rng
        self._nr = int(nr)
        self._center = center
        for k, v in kwargs.iteritems():
            v0 = getattr(self, k, None)
            assert not k.startswith('_') and v0 is not None
            setattr(self, k, type(v0)(v))

    def __call__(self):
        rst = self._gen_rot(self.min_angle, self.max_angle)
        if self.min_scale < self.max_scale:
            rst = self._update_scale(rst)
        rst = self._update_center(rst)
        return rst

    def _update_scale(self, m):
        r = lambda: self._rng.uniform(low=self.min_scale, high=self.max_scale)
        for i, j in zip(m, self._gen_rot(0, np.pi)):
            rot = j[:, :3]
            s = np.zeros_like(rot)
            s[0, 0] = r()
            s[1, 1] = r()
            s[2, 2] = r()
            i[:, :3] = rot.dot(s).dot(rot.T).dot(i[:, :3])
        return m

    def _update_center(self, m):
        if not self._center:
            return
        x, y, z = map(float, self._center)
        trans0 = np.eye(4)[:3]
        trans1 = trans0.copy()
        trans0[:3, 3] = -x, -y, -z
        trans1[:3, 3] = x, y, z
        for i in m:
            i[:] = afdot(trans1, i, trans0)
        return m

    def _gen_rot(self, min_angle, max_angle):
        # sample uniformly on sphere, rotation axis
        rv = self._rng.normal(size=(self._nr, 3))
        rv /= np.sqrt(np.square(rv).sum(axis=1, keepdims=True))

        # angle
        phi = self._rng.uniform(
            low=min_angle, high=max_angle, size=(self._nr, ))

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
             [m31, m32, m33, zero]]

        return np.transpose(m, axes=(2, 0, 1))
