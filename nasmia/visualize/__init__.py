# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Tue Apr 21 23:02:52 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .simple import SimpleData3DViewer

import numpy as np

def view_3d_data_simple(data, *args, **kwargs):
    SimpleData3DViewer(data, *args, **kwargs).mainloop()

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
