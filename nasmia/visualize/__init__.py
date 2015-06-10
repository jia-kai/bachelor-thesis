# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Wed Jun 10 23:28:55 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .single3d import Single3DDataViewer
from . import disp2d

import numpy as np

import logging
logger = logging.getLogger(__name__)

def view_3d_data_single(data, *args, **kwargs):
    Single3DDataViewer(data, *args, **kwargs).mainloop()

def view_3d_data_batched(data, individual_normalize=False, no_normalize=False,
                         **kwargs):
    if individual_normalize or no_normalize:
        assert not individual_normalize or not no_normalize
        if individual_normalize:
            d_tr = data.reshape((data.shape[0], -1))
            tshp = (data.shape[0], ) + (1, ) * (data.ndim - 1)
            d_max = np.max(d_tr, axis=1).reshape(tshp)
            d_min = np.min(d_tr, axis=1).reshape(tshp)
            dnorm = (data - d_min) / (d_max - d_min + 1e-9) * 255
        else:
            dnorm = np.clip(data, 0, 255)
        if data.ndim == 4:
            dslice = dnorm[:, data.shape[1] / 2]
        else:
            assert data.ndim == 5 and data.shape[1] in (1, 3)
            dslice = dnorm[:, :, data.shape[2] / 2]
    else:
        if data.ndim == 4:
            dslice = disp2d.normalize(data[:, data.shape[1] / 2])
        else:
            # color image
            assert data.ndim == 5 and data.shape[1] in (1, 3)
            dslice = disp2d.normalize(data[:, :, data.shape[2] / 2])

    def on_mouse(idx, _):
        logger.info('click on patch #{}'.format(idx))
        view_3d_data_single(data[idx], waitkey=False)

    disp2d.display_patch_list_2d(dslice, on_click=on_mouse, **kwargs)

def draw_box_on_image(image, pmin, pmax, color=None):
    if color is None:
        color = (0, 0, image.max())
    if image.ndim == 3:
        image = image[:,  :, :, np.newaxis]
        image = np.concatenate((image, image, image), axis=3)
    else:
        assert image.ndim == 4 and image.shape[3] == 3
    x0, y0, z0 = pmin
    x1, y1, z1 = pmax
    image[x0, y0:y1, z0:z1] = color
    image[x1, y0:y1, z0:z1] = color
    image[x0:x1, y0, z0:z1] = color
    image[x0:x1, y1, z0:z1] = color
    image[x0:x1, y0:y1, z0] = color
    image[x0:x1, y0:y1, z1] = color
    return image
