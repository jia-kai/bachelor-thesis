# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Sun Apr 19 21:45:37 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .simple import SimpleData3DViewer

def view_3d_data_simple(data, *args, **kwargs):
    SimpleData3DViewer(data, *args, **kwargs).mainloop()
