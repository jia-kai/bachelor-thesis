# -*- coding: utf-8 -*-
# $File: serial.py
# $Date: Sat Mar 28 22:45:26 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import joblib

def load(fpath, require_type=None):
    assert isinstance(fpath, basestring)
    obj = joblib.load(fpath)
    if require_type is not None:
        assert isinstance(obj, require_type), \
            '{} from {} is not an instance of {}'.format(
                type(obj), fpath, require_type)
    return obj


def dump(obj, fpath):
    """dump to file
    :param fpath: file obj or file path"""
    assert isinstance(fpath, basestring)
    joblib.dump(obj, fpath)
