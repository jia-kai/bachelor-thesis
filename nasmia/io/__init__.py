# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Sun May 31 19:15:55 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

class ModelEvalOutput(object):
    img = None
    """original input image"""

    ftr = None
    """model feature output"""

    def __init__(self, img, ftr):
        assert img.ndim == 3 and ftr.ndim == 4
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


class PointMatchResult(object):
    ref_idx = None
    """selected points on reference image;
    shape: (nr_point, 3)"""

    idx = None
    """matched points on test image;
    shape: (nr_point, 3)"""

    dist = None
    """matching dist in test image;
    shape: (nr_point, )"""

    img_shape = None
    """original test image shape"""

    args = None
    """original command line args passed to get_knn"""

    def __init__(self, ref_idx, idx, dist, img_shape, args):
        self.ref_idx = ref_idx
        self.idx = idx
        self.dist = dist
        self.img_shape = img_shape
        self.args = args


class TrainingData(object):
    patch = None
    """the image patches"""

    args = None
    """args passed to data cropper"""

    def __init__(self, patch, args):
        self.patch = patch
        self.args = args
