#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: get_match.py
# $Date: Sun May 31 19:19:32 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.math.op import sharedX
from nasmia.io import ModelEvalOutput, PointMatchResult
from nasmia.utils import serial, timed_operation, ProgressReporter

import numpy as np
import theano.tensor as T
import theano

import logging
import argparse
import itertools
from abc import abstractmethod, ABCMeta
logger = logging.getLogger(__name__)

class DistMeasure(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _do_calc(self, v0, v1):
        """:param v0: shape: (nr_sample0, feature_dim)
        :param v1: shape: (feature_dim, nr_sample1)"""
        pass

    def __call__(self, v0, v1):
        """:param v0, v1: feature maps, shape: (nr_sample, feature_dim)
        :return: pairwise dist of shape (nr_sample0, nr_sample1)"""
        assert v0.ndim == 2 and v1.ndim == 2 and v0.shape[1] == v1.shape[1]
        v0inp = v0.astype('float32')
        v1inp = v1.T.astype('float32')
        rst = self._do_calc(v0inp, v1inp)
        assert rst.ndim == 2 and rst.shape == (v0.shape[0], v1.shape[0]), \
            (rst.shape, v0.shape, v1.shape, self.__class__)
        return rst

    @abstractmethod
    def dist_brouteforce(self, v0, v1):
        """unoptimized dist, for testing"""


class L2Dist(DistMeasure):
    _func = None
    _do_sqrt = False
    def __init__(self, do_sqrt=False):
        self._do_sqrt = do_sqrt

    def _do_calc(self, v0, v1):
        if self._func is None:
            xv0 = T.matrix('v0')
            xv1 = T.matrix('v1')
            sqrsum0 = T.square(xv0).sum(axis=1, keepdims=True)
            sqrsum1 = T.square(xv1).sum(axis=0, keepdims=True)
            dot = T.dot(xv0, xv1)
            dist = sqrsum0 + sqrsum1 - dot * 2
            if self._do_sqrt:
                dist = T.sqrt(dist)
            self._func = theano.function([xv0, xv1], dist)
        return self._func(v0, v1)

    def dist_brouteforce(self, v0, v1):
        dist = np.square(v0 - v1).sum()
        if self._do_sqrt:
            dist = np.sqrt(dist)
        return dist


class CosDist(DistMeasure):
    _v0 = None
    _v1 = None
    _func = None

    def _do_calc(self, v0, v1):
        if self._func is None:
            xv0 = T.matrix('v0')
            xv1 = T.matrix('v1')
            norm0 = T.sqrt(T.square(xv0).sum(axis=1, keepdims=True))
            norm1 = T.sqrt(T.square(xv1).sum(axis=0, keepdims=True))
            dist = 1 - T.dot(xv0 / norm0, xv1 / norm1)
            self._func = theano.function([xv0, xv1], dist)
        return self._func(v0, v1)

    def dist_brouteforce(self, v0, v1):
        v0 = v0 / np.sqrt(np.square(v0).sum())
        v1 = v1 / np.sqrt(np.square(v1).sum())
        return 1 - np.dot(v0, v1)


class GetKNN(object):
    _args = None
    _rng = None

    def __init__(self, args):
        self._args = args
        self._rng = np.random.RandomState(args.seed)

        ref_ftr, ref_ftr_idx = self._load_ref_ftr()

        test_pack = serial.load(args.test, ModelEvalOutput)
        test_ftr = test_pack.ftr[
            :, ::args.test_downsample, ::args.test_downsample,
            ::args.test_downsample]

        ftr_dim = ref_ftr.shape[0]
        assert ftr_dim == test_ftr.shape[0]

        test_ftr_shape = test_ftr.shape[1:]
        ref_ftr = ref_ftr.reshape(ftr_dim, -1).T
        test_ftr = test_ftr.reshape(ftr_dim, -1).T

        raw_idx, match_dist = self._get_match_idx_and_dist(ref_ftr, test_ftr)
        match_dist = np.asarray(match_dist, dtype=np.float32)
        match_idx = [self._cvt_index_to_coord(i, test_ftr_shape)
                     for i in raw_idx]
        match_idx = np.asarray(match_idx, dtype=np.int32).reshape(
            ref_ftr.shape[0], 3)

        match_idx *= args.test_downsample
        match_idx -= test_pack.img2ftr_offset

        serial.dump(
            PointMatchResult(
                ref_idx=ref_ftr_idx,
                idx=match_idx, dist=match_dist, img_shape=test_pack.img.shape,
                args=args),
            args.output, use_pickle=True)

    def _get_match_idx_and_dist(self, ref_ftr, test_ftr):
        grp_dist = []
        grp_idx = []

        dist_measure = self._get_dist_measure()
        arng = np.arange(ref_ftr.shape[0])

        grp_idx_start = range(0, test_ftr.shape[0], self._args.batch_size)
        prog = ProgressReporter('eval', len(grp_idx_start))
        for i in grp_idx_start:
            pairwise_dist = dist_measure(
                ref_ftr, test_ftr[i:i+self._args.batch_size])
            cur_idx = np.argmin(pairwise_dist, axis=1)
            cur_dist = pairwise_dist[arng, cur_idx]
            cur_idx += i
            grp_idx.append(np.expand_dims(cur_idx, axis=1))
            grp_dist.append(np.expand_dims(cur_dist, axis=1))
            prog.trigger()
        prog.finish()

        grp_dist = np.concatenate(grp_dist, axis=1)
        grp_idx = np.concatenate(grp_idx, axis=1)

        sel_idx = np.argmin(grp_dist, axis=1)

        return grp_idx[arng, sel_idx], grp_dist[arng, sel_idx]

    @classmethod
    def _cvt_index_to_coord(cls, idx, shape):
        idx0 = idx
        rst = []
        for i in shape[::-1]:
            rst.append(idx % i)
            idx /= i
        assert idx == 0, 'index out of image: idx={} shape={}'.format(
            idx0, shape)
        return rst[::-1]

    def _get_dist_measure(self):
        m = self._args.measure
        if m == 'l2':
            return L2Dist()
        if m == 'cos':
            return CosDist()
        raise RuntimeError('unknown dist measure: {}'.format(m))

    def _load_ref_ftr(self):
        args = self._args

        ref = serial.load(args.ref, ModelEvalOutput)
        border_dist = serial.load(args.border, np.ndarray)

        assert border_dist.ndim == 3 and ref.img.shape == border_dist.shape

        mask = border_dist == args.ref_dist
        orig_nr_pt = mask.sum()
        mask = self._downsample_sparse_mask(mask, args.ref_downsample)
        ptlist = np.transpose(np.nonzero(mask))

        logger.info('number of reference points selected by dist: {}/{}'.format(
            ptlist.shape[0], orig_nr_pt))

        # only select those points in feature boundary
        in_bound_ = lambda v, d: ((v >= -ref.img2ftr_offset) *
                                 (v < ref.img.shape[d] + ref.img2ftr_offset))
        in_bound = lambda d: in_bound_(ptlist[:, d], d)
        ptlist_in_bound = in_bound(0) * in_bound(1) * in_bound(2)
        if ptlist_in_bound.sum() < ptlist.shape[0]:
            idx, = np.nonzero(ptlist_in_bound)
            logger.warn('only {} points in feature boundary; '
                        '{} discarded'.format(
                            len(idx), ptlist.shape[0] - len(idx)))
            ptlist = ptlist[idx]

        if args.nr_sample and ptlist.shape[0] > args.nr_sample:
            logger.info('random sample {} '
                        'feature points out of {}'.format(
                            args.nr_sample, len(ptlist)))
            self._rng.shuffle(ptlist)
            ptlist = ptlist[:args.nr_sample]

        logger.info('conv shape: {}'.format(ref.conv_shape))

        ptlist_orig = ptlist
        ptlist = ptlist_orig + ref.img2ftr_offset
        assert ptlist.min() >= 0
        idx0, idx1, idx2 = ptlist.T
        return ref.ftr[:, idx0, idx1, idx2], ptlist_orig

    def _downsample_sparse_mask(self, mask, factor):
        if factor <= 1:
            return mask

        logger.info('downsample mask factor: {}'.format(factor))
        assert mask.dtype == np.bool
        available = np.zeros_like(mask, dtype=np.int32)

        for i, j, k in itertools.product(range(factor), repeat=3):
            sub = mask[i::factor, j::factor, k::factor]
            avail_sub = available[:sub.shape[0], :sub.shape[1], :sub.shape[2]]
            avail_sub += sub

        available = ((available > 0) +
                     available * self._rng.uniform(size=available.shape))
        available = available.astype(np.int)

        for i, j, k in itertools.product(range(factor), repeat=3):
            sub = mask[i::factor, j::factor, k::factor]
            avail_sub = available[:sub.shape[0], :sub.shape[1], :sub.shape[2]]
            avail_sub_next = avail_sub - sub
            sub *= avail_sub == 1
            avail_sub[:] = avail_sub_next
        assert available.max() == 0
        return mask

def run_tests():
    nr0 = 100
    nr1 = 50
    ftrdim = 23
    v0 = np.random.normal(size=(nr0, ftrdim))
    v1 = np.random.normal(size=(nr1, ftrdim))
    for cls in DistMeasure.__subclasses__():
        measure = cls()
        dist = measure(v0, v1)
        max_err = 0
        for i in range(nr0):
            for j in range(nr1):
                got = dist[i, j]
                expect = measure.dist_brouteforce(v0[i], v1[j])
                err = abs(got - expect)
                max_err = max(max_err, err)
                assert err < 1e-4, (got, expect, cls)
        logger.info('{} max err: {}'.format(cls.__name__, max_err))


def main():
    parser = argparse.ArgumentParser(
        description='find points on test image that match with boundary points'
        'on ref image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # io
    parser.add_argument('-r', '--ref', required=True,
                        help='reference image with features')
    parser.add_argument('-b', '--border', required=True,
                        help='border dist of reference image')
    parser.add_argument('-t', '--test', required=True,
                        help='test image with features')
    parser.add_argument('-o', '--output', required=True,
                        help='output file path (save dict: idx, dist)')
    # ref options
    parser.add_argument('--ref_dist', type=int, default=2,
                        help='distance for points selected on reference image')
    parser.add_argument('--nr_sample', type=int,
                        help='number of points to sample on reference image')
    parser.add_argument('--ref_downsample', type=int, default=3,
                        help='downsample ratio of reference image')
    parser.add_argument('--test_downsample', type=int, default=1,
                        help='downsample ratio of test image')
    # match options
    parser.add_argument('--measure', choices=['l2', 'cos'], default='cos',
                        help='distance measure to use')
    # misc
    parser.add_argument('--run_tests', action='store_true',
                        help='run dist measure tests')
    parser.add_argument('--seed', type=int, default=20150511,
                        help='RNG seed')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='size of batch for a single run')
    args = parser.parse_args()
    if args.run_tests:
        run_tests()
        return

    GetKNN(args)


if __name__ == '__main__':
    main()
