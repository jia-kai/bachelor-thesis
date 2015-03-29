# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Sun Mar 29 10:02:07 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .simple import SimpleData3DViewer

def view_3d_data_simple(data, scale=1):
    SimpleData3DViewer(data, scale).mainloop()
