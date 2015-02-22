# -*- coding: utf-8 -*-
# $File: visualize.py
# $Date: Sun Feb 22 21:15:23 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import functools

import cv2

class Data3DViewer(object):
    x = 0
    y = 0
    z = 0

    def __init__(self, data):
        data = cv2.normalize(
            data,
            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')
        self.data = data

    def mainloop(self):
        self.x = self.y = self.z = 0
        for i in range(3):
            ax_name = chr(ord('x') + i)
            win_name = 'view_' + ax_name
            fshow = functools.partial(self._showimg, win_name=win_name, axis=i)
            fshow(0)
            cv2.createTrackbar(ax_name, win_name,
                               0, self.data.shape[i] - 1, fshow)
        while True:
            key = chr(cv2.waitKey(-1) & 0xFF)
            if key == 'q':
                return

    def _showimg(self, pos, win_name, axis):
        s = slice(None, None, None)
        ind = [(pos, ), (s, pos,), (s, s, pos)]
        img = self.data[ind[axis]]
        cv2.imshow(win_name, img)


def view_3d_data(data):
    Data3DViewer(data).mainloop()
