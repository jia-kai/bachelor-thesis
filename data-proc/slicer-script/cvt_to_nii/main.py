# -*- coding: utf-8 -*-
# $File: main.py
# $Date: Fri May 08 10:35:14 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

import os
import load_and_save

def run():
    flist = map('{:03d}'.format, range(1, 21))
    flist = (map('liver-orig{}'.format, flist) +
             map('liver-seg{}'.format, flist))
    for i in flist:
        inp = '/mnt/usbhd/jiakai/work/sliver07/train/{}.hdr'.format(i)
        opath = '/mnt/usbhd/jiakai/work/sliver07/train-converted/{}.nii.gz'.format(i)
        if os.path.isfile(opath):
            print '{} exists, skipped'.format(opath)
            continue
        print 'working on', inp
        load_and_save.run(inp, opath)
