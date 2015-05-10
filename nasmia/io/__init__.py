# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Sun May 10 22:17:40 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

class ModelEvalOutput(object):
    img = None
    """original input image"""

    ftr = None
    """model feature output"""

    def __init__(self, img, ftr):
        self.img = img
        self.ftr = ftr


class TrainingData(object):
    patch = None
    """the image patches"""

    args = None
    """args passed to data cropper"""

    def __init__(self, patch, args):
        self.patch = patch
        self.args = args
