# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Fri May 01 23:54:56 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .single3d import Single3DDataViewer
from . import disp2d

def view_3d_data_single(data, *args, **kwargs):
    Single3DDataViewer(data, *args, **kwargs).mainloop()

def view_3d_data_batched(data):
    dslice = disp2d.normalize(data[:, data.shape[1] / 2])
    def on_mouse(idx, _):
        view_3d_data_single(data[idx], waitkey=False)

    disp2d.display_patch_list_2d(dslice, on_click=on_mouse)
