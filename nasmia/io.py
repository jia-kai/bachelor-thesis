# -*- coding: utf-8 -*-
# $File: io.py
# $Date: Sat Mar 07 16:48:31 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np

import os
import logging
from collections import OrderedDict
logger = logging.getLogger(__name__)

class ScenePackReader(object):
    _fin_data = None
    _cur_obj = None

    scenes = None

    def __init__(self, fpath):
        self.scenes = []
        logger.info('load meta file {}'.format(fpath))
        with open(fpath) as fin:
            for line in fin:
                cmd, arg = line.strip().split(': ', 1)
                if self._cur_obj is None or cmd == 'end':
                    func = '_on_cmd_{}'.format(cmd)
                    getattr(self, func)(arg)
                else:
                    self._cur_obj.on_init_command(cmd, arg)

    def _on_cmd_fpath_data(self, fpath):
        assert self._fin_data is None
        logger.info('open data file {}'.format(fpath))
        self._fin_data = open(fpath)

    def _on_cmd_file(self, fpath):
        self.scenes.append(SceneReader(fpath))

    def _on_cmd_image(self, name):
        self._cur_obj = ImageReader(name, self._fin_data)
        self.scenes[-1].add_object(self._cur_obj)

    def _on_cmd_surface(self, name):
        self._cur_obj = SurfaceReader(name, self._fin_data)
        self.scenes[-1].add_object(self._cur_obj)

    def _on_cmd_end(self, name):
        assert name == self._cur_obj.name
        self._cur_obj = None


class SceneReader(object):
    fpath = None
    objects = None

    def __init__(self, fpath):
        self.fpath = fpath
        self.objects = OrderedDict()

    def add_object(self, obj):
        name = obj.name
        appended = -1
        while name in self.objects:
            logger.warn('duplicated object name: {}'.format(name))
            if appended == -1:
                name += '-0'
                appended = 0
            else:
                appended += 1
                name = '{}-{}'.format(obj.name, appended)
        self.objects[obj.name] = obj
        self.objects[len(self.objects) / 2] = obj

    def __str__(self):
        return 'SceneReader(file={}, nr_obj={})'.format(
            os.path.basename(self.fpath), len(self.objects))

    def __repr__(self):
        return str(self)


class ImageReader(object):
    name = None
    shape = None
    spacing = None
    origin = None
    dtype = None
    _offset = None
    _fin = None

    _data_cache = None

    def __init__(self, name, file_input):
        self.name = name
        self._fin = file_input

    def on_init_command(self, name, args):
        trait = {
            'dim': (int, 'shape'),
            'spacing': (float, 'spacing'),
            'origin': (float, 'origin')
        }
        if name in trait:
            dt, attr = trait[name]
            x, y, z = map(dt, args.split())
            assert getattr(self, attr) is None
            setattr(self, attr, (x, y, z))
            return
        if name == 'pixel_type':
            self.dtype = args
            return
        if name == 'offset':
            self._offset = int(args)
            return
        raise ValueError('unknown image prop: {}'.format(name))

    @property
    def data(self):
        """data as :class:`numpy.ndarray`"""
        if self._data_cache is not None:
            return self._data_cache
        self._fin.seek(self._offset)
        data = np.fromfile(
            self._fin, dtype=self.dtype, count=np.prod(self.shape)).reshape(
                self.shape[::-1])
        data = data.transpose((2, 1, 0))
        self._data_cache = data
        del self._fin
        return data

    def __str__(self):
        return 'ImageReader(name={}, shape={}, dtype={})'.format(
            self.name, self.shape, self.dtype)

    def __repr__(self):
        return str(self)


class SurfaceReader(object):
    name = None

    _fin = None
    _offset = None
    _nr_pt = None
    _nr_triangle = None

    _data_cache = None

    def __init__(self, name, file_input):
        self.name = name
        self._fin = file_input

    def on_init_command(self, name, args):
        if name == 'num_point_triangle':
            self._nr_pt, self._nr_triangle = map(
                int, args.split())
            return
        assert name == 'offset'
        self._offset = int(args)

    def _load_data(self):
        if self._data_cache is not None:
            return self._data_cache

        rd = lambda dtype, nr: np.fromfile(
            self._fin, dtype=dtype, count=nr * 3).reshape(nr, 3)
        self._fin.seek(self._offset)
        points = rd('float32', self._nr_pt)
        tri = rd('int32', self._nr_triangle)
        self._data_cache = (points, tri)
        del self._fin
        return self._data_cache

    @property
    def points(self):
        """ndarray of shape (x, 3) to describe the points"""
        return self._load_data()[0]

    @property
    def triangles(self):
        """ndarray of shape (x, 3) to describe the points"""
        return self._load_data()[1]

    def __str__(self):
        return 'SurfaceReader(name={}, nr_point={}, nr_triangle={})'.format(
            self.name, self._nr_pt, self._nr_triangle)

    def __repr__(self):
        return str(self)
