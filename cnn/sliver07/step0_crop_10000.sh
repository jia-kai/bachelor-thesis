#!/bin/bash -e
# $File: step0_crop_10000.sh
# $Date: Sat May 23 12:15:29 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data
../crop_patch.py ../../../sliver07/list.txt data/train-10000perimg.pkl \
    --patch_per_img 10000
