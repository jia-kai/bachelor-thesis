#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: disp2d.py
# $Date: Sat May 02 12:06:44 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import cv2
import numpy as np

import sys
import logging
logger = logging.getLogger()

def minnone(x, y):
    if x is None:
        x = y
    elif y is None:
        y = x
    return min(x, y)

def display_patch_list_2d(plist, nr_row=None, nr_col=None, sep=1,
                       max_width=600, max_height=600,
                       win_title='patch', on_click=None,
                       on_page_change=None, destroy_window=True):
    """:param plist: patch list, of shape ('b', 'c', 0, 1)
    :param on_click: callback taking (idx, patch) as argument
    :param on_page_change: callback taking (idx_start, idx_end, patch_sub) as
        argument
    """
    if on_click is None:
        on_click = lambda idx, patch: logger.info('click on {}'.format(idx))

    assert plist.ndim == 3
    logger.info('display patch, shape={}'.format(plist.shape))
    plist = plist.copy()
    plist_orig = plist.copy()
    idxarr = range(plist.shape[0])

    ph, pw = plist.shape[1:]
    nr_row = minnone(nr_row, max_height / (ph + sep))
    nr_col = minnone(nr_col, max_width / (pw + sep))

    img = np.zeros(
        (nr_row * (ph + sep) - sep, nr_col * (pw + sep) - sep), dtype='uint8')

    def do_show(plist):
        if not len(plist):
            return
        cur_row = 0
        cur_col = 0
        img.fill(0)
        for patch in plist:
            r0 = cur_row * (ph + sep)
            c0 = cur_col * (pw + sep)
            img[r0:r0+ph, c0:c0+pw] = patch
            cur_col += 1
            if cur_col == nr_col:
                cur_col = 0
                cur_row += 1
        cv2.imshow(win_title, img)
        cv2.setMouseCallback(win_title, on_mouse)


    nr_patch_per_page = nr_row * nr_col
    def on_mouse(event, x, y, *args):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x /= pw + sep
        y /= ph + sep
        idx = start + y * nr_col + x
        on_click(idxarr[idx], plist[idx])

    start = 0
    while True:
        end = start + nr_patch_per_page
        cur_plist = plist[start:end]
        end = start + cur_plist.shape[0]
        logger.info('showing patches {} to {}'.format(start, end))
        if on_page_change is not None:
            on_page_change(start, end, cur_plist)
        do_show(cur_plist)
        while True:
            key = chr(cv2.waitKey(-1) & 0xFF)

            if key == 'q':
                if destroy_window:
                    cv2.destroyWindow(win_title)
                return
            if key == 'h' and start:
                start -= nr_patch_per_page
                break
            if key == 'l' and end < plist.shape[0]:
                start = end
                break
            if key == 's':
                logger.info('shuffle all')
                start = 0
                np.random.shuffle(idxarr)
                plist = plist_orig[idxarr]
                break
            if key == 'x':
                logger.warn('x pressed, exit')
                sys.exit(5)

def normalize(data):
    """linearly scale to [0, 255]"""
    assert data.ndim == 3
    #d_tr = data.reshape((data.shape[0], -1))
    #d_max = np.max(d_tr, axis=1).reshape((data.shape[0], 1, 1))
    #d_min = np.min(d_tr, axis=1).reshape((data.shape[0], 1, 1))
    d_min = data.min()
    d_max = data.max()
    return (data - d_min) / (d_max - d_min + 1e-9) * 255
