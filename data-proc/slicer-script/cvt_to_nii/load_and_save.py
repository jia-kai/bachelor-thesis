#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: load_and_save.py
# $Date: Thu May 07 22:59:32 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import slicer
from slicer.util import getNode

import os
import gc

_recent_opr_nodes = []

def _load_volume(path, name=None):
    succ, node = slicer.util.loadVolume(path, returnNode=True)
    assert succ is True
    if name:
        node.SetName(name)
    return node

def run(input_fpath, output_fpath, input_node_name=None):
    assert _recent_opr_nodes == []
    input_node = None
    if input_node_name is not None:
        input_node = getNode(input_node_name)
    if input_node is None:
        input_node = _load_volume(input_fpath, input_node_name)

    slicer.util.saveNode(input_node, output_fpath + '.nii.gz')
