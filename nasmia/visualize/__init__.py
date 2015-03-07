# -*- coding: utf-8 -*-
# $File: __init__.py
# $Date: Sat Mar 07 15:58:49 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

from .simple import SimpleData3DViewer

def view_3d_data_simple(data):
    SimpleData3DViewer(data).mainloop()
