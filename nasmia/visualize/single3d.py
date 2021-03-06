# -*- coding: utf-8 -*-
# $File: single3d.py
# $Date: Sun Jun 07 22:40:11 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import numpy as np
import cv2

import functools
import sys
import logging
logger = logging.getLogger(__name__)

class Single3DDataViewer(object):
    data = None
    scale = None
    _waitkey = None
    _onclick = None
    _axis_pos = None
    _onaxischange = None
    _prefix=None

    def __init__(self, data, scale=None, onclick=None, onaxischange=None,
                 waitkey=True, prefix=''):
        """:param onclick: callback, (x, y, z)
        :param onaxischange: callback, (axis, pos)"""
        assert data.ndim in (3, 4)
        if data.ndim == 4:
            assert data.shape[0] in (1, 3)
            data = np.transpose(data, (1, 2, 3, 0))
        self._axis_pos = [0] * 3
        dmin = data.min()
        dmax = data.max()
        self.data = ((data - dmin) *
                     (255.0 / (dmax - dmin + 1e-9))).astype('uint8')
        if scale is None:
            scale = 400 / np.max(data.shape)
        self.scale = int(max(scale, 1))
        self._onclick = onclick
        self._onaxischange = onaxischange
        self._waitkey = waitkey
        self._prefix = prefix

    def mainloop(self):
        for i in range(3):
            ax_name = chr(ord('x') + i)
            win_name = self._prefix + 'view_' + ax_name
            fshow = functools.partial(self._showimg, win_name=win_name, axis=i)
            pos = self.data.shape[i] / 2
            fshow(pos)
            if self.data.shape[i] > 1:
                cv2.createTrackbar(ax_name, win_name,
                                   pos, self.data.shape[i] - 1, fshow)
            cv2.setMouseCallback(
                win_name,
                functools.partial(self._on_mouse, axis=i))

        while self._waitkey:
            key = chr(cv2.waitKey(-1) & 0xFF)
            if key == 'x':
                logger.info('x pressed, exit')
                sys.exit()
            if key == 'q':
                return

    def _showimg(self, pos, win_name, axis):
        self._axis_pos[axis] = pos
        s = slice(None, None, None)
        ind = [(pos, ), (s, pos,), (s, s, pos)][axis]
        img = self.data[ind]
        if self.scale != 1:
            img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow(win_name, img)
        if self._onaxischange:
            self._onaxischange(axis, pos)

    def _on_mouse(self, event, y, x, *args, **kwargs):
        if self._onclick is None or event != cv2.EVENT_LBUTTONDOWN:
            return
        x /= self.scale
        y /= self.scale
        axis = kwargs['axis']
        coord = [x, y]
        coord.insert(axis, self._axis_pos[axis])
        self._onclick(*coord)
