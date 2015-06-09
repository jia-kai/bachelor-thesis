#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: make_gamma.py
# $Date: Fri Jun 05 18:09:18 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import cv2
import numpy as np

def gamma(img, r):
    outv = (np.power(img / 255.0, r) * 255).astype(np.uint8)
    cv2.putText(outv, 'gamma={}'.format(r if r != 1 else '1 (original)'),
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return outv

img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

out_img = np.zeros((img.shape[0], img.shape[1] * 3 + 2), dtype=np.uint8)
xpos = [0]

def do(r):
    x = xpos[0]
    xpos[0] += img.shape[1] + 1
    out_img[:, x:x+img.shape[1]] = gamma(img, r)

do(0.5)
do(1.0)
do(2.0)

cv2.imwrite('gamma.png', out_img)
