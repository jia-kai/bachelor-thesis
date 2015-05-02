# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Fri May 01 11:37:30 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import logging
import sys
import os
logger = logging.getLogger(__name__)

def sharedX(x, name=None):
    """shared theano.floatX"""
    import theano
    x = theano.shared(
        theano._asarray(x, dtype=theano.config.floatX),
        name=name)
    return x

def floatX():
    return 'float32'
    #import theano
    #return theano.config.floatX

def set_gpu_num(num):
    assert 'theano' not in sys.modules
    if num is None:
        return
    orig_flag = os.getenv('THEANO_FLAGS', '').split(';')
    flag = [i for i in orig_flag if i.strip() and not i.startswith('device=')]
    flag.append('device=gpu{}'.format(num))
    os.environ['THEANO_FLAGS'] = ';'.join(flag)
    logger.info('set THEANO_FLAGS: {}'.format(os.environ['THEANO_FLAGS']))
