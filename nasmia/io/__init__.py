# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Mon May 11 21:56:13 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

class ModelEvalOutput(object):
    img = None
    """original input image"""

    ftr = None
    """model feature output"""

    def __init__(self, img, ftr):
        self.img = img
        self.ftr = ftr


    @property
    def conv_shape(self):
        rst = None
        for i in range(3):
            s = self.img.shape[i] + 1 - self.ftr.shape[i + 1]
            if rst is None:
                rst = s
            else:
                assert rst == s
        return rst

    @property
    def img2ftr_offset(self):
        """offset to convert image coordinate to feature coordinate"""
        return -self.conv_shape / 2


class KNNResult(object):
    idx = None
    """shape: (nr_point, nr_knn, 3)"""

    dist = None
    """shape: (nr_point, nr_knn)"""

    img_shape = None
    """original test image shape"""

    def __init__(self, idx, dist, img_shape):
        self.idx = idx
        self.dist = dist
        self.img_shape = img_shape


class TrainingData(object):
    patch = None
    """the image patches"""

    args = None
    """args passed to data cropper"""

    def __init__(self, patch, args):
        self.patch = patch
        self.args = args
