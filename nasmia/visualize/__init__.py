# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Sun Jun 07 22:43:31 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .single3d import Single3DDataViewer
from . import disp2d

import numpy as np

def view_3d_data_single(data, *args, **kwargs):
    Single3DDataViewer(data, *args, **kwargs).mainloop()

def view_3d_data_batched(data, **kwargs):
    if data.ndim == 4:
        dslice = disp2d.normalize(data[:, data.shape[1] / 2])
    else:
        # color image
        assert data.ndim == 5 and data.shape[1] in (1, 3)
        dslice = disp2d.normalize(data[:, :, data.shape[2] / 2])
    def on_mouse(idx, _):
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
