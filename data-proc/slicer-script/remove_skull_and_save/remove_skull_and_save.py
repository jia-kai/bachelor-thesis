#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: remove_skull_and_save.py
# $Date: Sat Mar 21 20:03:49 2015 +0800
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

def _on_finished():
    if not _recent_opr_nodes:
        print('already saved, completed twice? WTF?!')
        return
    print('finished, save output')
    inp, out_volume, out_mask, out_path = _recent_opr_nodes
    _recent_opr_nodes[:] = []

    slicer.util.saveNode(inp, out_path + '-orig.nrrd')
    slicer.util.saveNode(out_mask, out_path + '-mask.nrrd')
    print('output saved: {}'.format(out_path))

    def destory(node):
        slicer.mrmlScene.RemoveNode(node)
        #node.UnRegister(slicer.mrmlScene)
        node.GetImageData().ReleaseData()
    destory(inp)
    destory(out_volume)
    destory(out_mask)
    print('gc: {}'.format(gc.collect()))


def _on_status_modified(caller, event):
    print("Got a %s from a %s" % (event, caller.GetClassName()))
    if caller.IsA('vtkMRMLCommandLineModuleNode'):
        print("Status is %s" % caller.GetStatusString())
        if caller.GetStatus() == 0x20: # completed
            _on_finished()

def _get_atlas_nodes(volume_name='atlasImage',
                     mask_name='atlasMask',
                     volume_path='data/atlasImage.mha',
                     mask_path='data/atlasMask.mha'):

    volume_node = getNode(volume_name)
    mask_node = getNode(mask_name)
    if volume_node is not None and mask_node is not None:
        return volume_node, mask_node
    fname = __file__
    if fname.endswith('.pyc'):
        fname = fname[:-1]
    basedir = os.path.dirname(os.path.realpath(fname))
    p = lambda v: os.path.join(basedir, v)
    if volume_node is None:
        volume_node = _load_volume(p(volume_path), volume_name)

    if mask_node is None:
        mask_node = _load_volume(p(mask_path), mask_name)
        mask_node.LabelMapOn()
    return volume_node, mask_node


def run(input_fpath, output_fpath, input_node_name=None):
    assert _recent_opr_nodes == []
    input_node = None
    if input_node_name is not None:
        input_node = getNode(input_node_name)
    if input_node is None:
        input_node = _load_volume(input_fpath, input_node_name)

    atlas_volume, atlas_mask = _get_atlas_nodes()

    output_volume_node = slicer.vtkMRMLScalarVolumeNode()
    output_mask_node = slicer.vtkMRMLScalarVolumeNode()
    output_volume_node.SetName('{}-output'.format(input_node.GetName()))
    output_mask_node.SetName('{}-mask'.format(input_node.GetName()))
    slicer.mrmlScene.AddNode(output_volume_node)
    slicer.mrmlScene.AddNode(output_mask_node)
    output_mask_node.LabelMapOn()

    params = {
        'atlasMRIVolume': atlas_volume.GetID(),
        'atlasMaskVolume': atlas_mask.GetID(),
        'patientVolume': input_node.GetID(),
        'patientOutputVolume': output_volume_node.GetID(),
        'patientMaskLabel': output_mask_node.GetID()
    }

    cli_node = slicer.cli.run(slicer.modules.swissskullstripper, None, params,
                              wait_for_completion=True)
    # cli_node.AddObserver('ModifiedEvent', _on_status_modified)
    _recent_opr_nodes[:] = [
        input_node, output_volume_node, output_mask_node, output_fpath]
    _on_finished()
    #return cli_node
