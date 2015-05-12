#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: get_knn.py
# $Date: Tue May 12 16:35:00 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from nasmia.math.op import sharedX
from nasmia.io import ModelEvalOutput, KNNResult
from nasmia.utils import serial, timed_operation

import numpy as np
import theano.tensor as T
import theano

from abc import abstractmethod, ABCMeta
import logging
import argparse
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
        rst = self._do_calc(v0, v1.T)
        assert rst.ndim == 2 and rst.shape == (v0.shape[0], v1.shape[0]), \
            (rst.shape, v0.shape, v1.shape, self.__class__)
        return rst

    @abstractmethod
    def dist_brouteforce(self, v0, v1):
        """unoptimized dist, for testing"""


class L2Dist(DistMeasure):
    _do_sqrt = False
    def __init__(self, do_sqrt=False):
        self._do_sqrt = do_sqrt

    def _do_calc(self, v0, v1):
        v0_shp = v0.shape
        v1_shp = v1.shape
        v0 = sharedX(v0)
        v1 = sharedX(v1)
        sqrsum0 = T.square(v0).sum(axis=1, keepdims=True)
        sqrsum1 = T.square(v1).sum(axis=0, keepdims=True)
        dot = T.dot(v0, v1)
        dist = sqrsum0 + sqrsum1 - dot * 2
        if self._do_sqrt:
            dist = T.sqrt(dist)
        func = theano.function([], dist)
        with timed_operation('L2Dist: {}x{}'.format(
                v0_shp, v1_shp)):
            return func()

    def dist_brouteforce(self, v0, v1):
        dist = np.square(v0 - v1).sum()
        if self._do_sqrt:
            dist = np.sqrt(dist)
        return dist


class GetKNN(object):
    _args = None
    _rng = None

    def __init__(self, args):
        self._args = args
        self._rng = np.random.RandomState(args.seed)

        ref_ftr = self._load_ref_ftr()

        test_pack = serial.load(args.test, ModelEvalOutput)
        test_ftr = test_pack.ftr[
            :, ::args.test_downsample, ::args.test_downsample,
            ::args.test_downsample]

        ftr_dim = ref_ftr.shape[0]
        assert ftr_dim == test_ftr.shape[0]

        dist = self._get_dist_measure()(
            ref_ftr.reshape(ftr_dim, -1).T,
            test_ftr.reshape(ftr_dim, -1).T)

        knn_idx, knn_dist = self._get_all_knn_index(dist)
        knn_idx = [self._cvt_index_to_coord(i, test_ftr.shape[1:])
                   for i in knn_idx]
        knn_idx = np.array(knn_idx)
        knn_dist = np.array(knn_dist)

        knn_idx *= args.test_downsample
        knn_idx -= test_pack.img2ftr_offset
        knn_idx = knn_idx.reshape(ref_ftr.shape[1], args.nr_knn, 3)
        knn_dist = knn_dist.reshape(ref_ftr.shape[1], args.nr_knn)

        serial.dump(
            KNNResult(idx=knn_idx, dist=knn_dist,
                      img_shape=test_pack.img.shape,
                      args=args),
            args.output, use_pickle=True)

    @classmethod
    def _cvt_index_to_coord(cls, idx, shape):
        rst = []
        for i in shape[::-1]:
            rst.append(idx % i)
            idx /= i
        assert idx == 0
        return rst[::-1]

    def _get_all_knn_index(self, dist):
        with timed_operation('get_knn_indx_cpu: nr={}'.format(
                self._args.nr_knn)):
            return self._get_all_knn_index_cpu(dist)

    def _get_all_knn_index_cpu(self, dist):
        dist = dist.copy()
        rst_idx = []
        rst_dist = []

        shp0 = range(dist.shape[0])
        for i in range(self._args.nr_knn):
            idx = np.argmin(dist, axis=1)
            rst_idx.extend(idx)
            rst_dist.extend(dist[shp0, idx])
            dist[shp0, idx] = np.inf
        return rst_idx, rst_dist

    def _get_all_knn_index_gpu(self, dist):
        shp0 = T.arange(dist.shape[0])
        dist = sharedX(dist)

        rst_idx = []
        rst_dist = []

        for i in range(self._args.nr_knn):
            idx = T.argmin(dist, axis=1)
            dsub = dist[shp0, idx]
            rst_idx.append(idx)
            rst_dist.append(dsub)
            dist = T.set_subtensor(dsub, np.inf)

        rst_idx = T.concatenate(rst_idx, axis=0)
        rst_dist = T.concatenate(rst_dist, axis=0)
        func = theano.function([], [rst_idx, rst_dist])
        with timed_operation('get knns'):
            return func()

    def _get_dist_measure(self):
        m = self._args.measure
        if m == 'l2':
            return L2Dist()

        raise RuntimeError('unknown dist measure: {}'.format(m))

    def _load_ref_ftr(self):
        args = self._args

        ref = serial.load(args.ref, ModelEvalOutput)
        border_dist = serial.load(args.border, np.ndarray)

        assert border_dist.ndim == 3

        ptlist = np.transpose(np.nonzero(border_dist == args.ref_dist))

        logger.info('number of reference points selected by dist: {}'.format(
            ptlist.shape[0]))
        if ptlist.shape[0] > args.nr_sample:
            self._rng.shuffle(ptlist)
            ptlist = ptlist[:args.nr_sample]

        logger.info('conv shape: {}'.format(ref.conv_shape))

        ptlist += ref.img2ftr_offset
        assert ptlist.min() >= 0
        idx0, idx1, idx2 = ptlist.T
        return ref.ftr[:, idx0, idx1, idx2]


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
        description='calc KNN of boundary points and save to file',
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
    parser.add_argument('--nr_sample', type=int, default=1000,
                        help='number of points to sample on reference image')
    # test options
    parser.add_argument('--test_downsample', type=int, default=3,
                        help='downsample ratio of test image')
    # knn options
    parser.add_argument('--measure', choices=['l2'], default='l2',
                        help='distance measure to use')
    parser.add_argument('--nr_knn', type=int, default=1,
                        help='number of KNNs to return')
    # misc
    parser.add_argument('--run_tests', action='store_true',
                        help='run dist measure tests')
    parser.add_argument('--seed', type=int, default=20150511,
                        help='RNG seed')
    args = parser.parse_args()
    if args.run_tests:
        run_tests()

    GetKNN(args)


if __name__ == '__main__':
    main()
