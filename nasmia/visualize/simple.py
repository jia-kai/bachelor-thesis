# -*- coding: utf-8 -*-
# $File: simple.py
# $Date: Sun Mar 29 10:01:52 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import functools

import cv2

class SimpleData3DViewer(object):
    data = None
    scale = None

    def __init__(self, data, scale=1):
        dmin = data.min()
        dmax = data.max()
        self.data = ((data - dmin) * (255.0 / (dmax - dmin))).astype('uint8')
        self.scale = scale

    def mainloop(self):
        for i in range(3):
            ax_name = chr(ord('x') + i)
            win_name = 'view_' + ax_name
            fshow = functools.partial(self._showimg, win_name=win_name, axis=i)
            pos = self.data.shape[i] / 2 if i != 2 else 0
            fshow(pos)
            cv2.createTrackbar(ax_name, win_name,
                               pos, self.data.shape[i] - 1, fshow)
        while True:
            key = chr(cv2.waitKey(-1) & 0xFF)
            if key == 'q':
                return

    def _showimg(self, pos, win_name, axis):
        s = slice(None, None, None)
        ind = [(pos, ), (s, pos,), (s, s, pos)]
        img = self.data[ind[axis]]
        if self.scale != 1:
            img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow(win_name, img)
