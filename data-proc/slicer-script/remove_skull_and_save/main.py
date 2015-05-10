# -*- coding: utf-8 -*-
# $File: main.py
# $Date: Sat Mar 21 17:27:50 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import os
import remove_skull_and_save

def run():
    flist = map('{:04d}'.format, range(41))
    for i in flist:
        opath = '/mnt/usbhd/jiakai/work/ADNI/converted/{}'.format(i)
        if os.path.isfile(opath + '-mask.nrrd'):
            print '{} exists, skipped'.format(opath)
            continue
        remove_skull_and_save.run(
            '/mnt/usbhd/jiakai/work/ADNI/orig/all/{}.nii'.format(i),
            opath, 'img{}'.format(i))
