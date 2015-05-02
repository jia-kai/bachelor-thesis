# -*- coding: utf-8 -*-
# $File: simple.py
# $Date: Tue Apr 21 23:03:53 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import functools

import cv2

class SimpleData3DViewer(object):
    data = None
    scale = None
    _waitkey = None
    _onclick = None
    _axis_pos = None
    _onaxischange = None
    _prefix=None

    def __init__(self, data, scale=1, onclick=None, onaxischange=None,
                 waitkey=True, prefix=''):
        """:param onclick: callback, (x, y, z)
        :param onaxischange: callback, (axis, pos)"""
        assert data.ndim in (3, 4)
        self._axis_pos = [0] * 3
        dmin = data.min()
        dmax = data.max()
        self.data = ((data - dmin) * (255.0 / (dmax - dmin))).astype('uint8')
        self.scale = scale
        self._onclick = onclick
        self._onaxischange = onaxischange
        self._waitkey = waitkey
        self._prefix = prefix

    def mainloop(self):
        for i in range(3):
            ax_name = chr(ord('x') + i)
            win_name = self._prefix + 'view_' + ax_name
            fshow = functools.partial(self._showimg, win_name=win_name, axis=i)
            pos = self.data.shape[i] / 2 if i != 2 else 0
            fshow(pos)
            cv2.createTrackbar(ax_name, win_name,
                               pos, self.data.shape[i] - 1, fshow)
            cv2.setMouseCallback(
                win_name,
                functools.partial(self._on_mouse, axis=i))

        while self._waitkey:
            key = chr(cv2.waitKey(-1) & 0xFF)
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
        axis = kwargs['axis']
        if self._onclick is None or event != cv2.EVENT_LBUTTONDOWN:
            return
        coord = [x, y]
        coord.insert(axis, self._axis_pos[axis])
        self._onclick(*coord)
