# -*- coding: utf-8 -*-
# $File: serial.py
# $Date: Sun Mar 29 09:31:25 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import joblib
import cPickle as pickle

def load(fpath, require_type=None):
    assert isinstance(fpath, basestring)
    try:
        obj = joblib.load(fpath)
    except:
        with open(fpath) as fin:
            obj = pickle.load(fin)
    if require_type is not None:
        assert isinstance(obj, require_type), \
            '{} from {} is not an instance of {}'.format(
                type(obj), fpath, require_type)
    return obj


def dump(obj, fpath, use_pickle=False):
    """dump to file
    :param fpath: file obj or file path"""
    assert isinstance(fpath, basestring)
    if use_pickle:
        with open(fpath, 'wb') as fout:
            pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)
    else:
        joblib.dump(obj, fpath)
