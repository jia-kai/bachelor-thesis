#!/bin/bash -e
# $File: step0_crop.sh
# $Date: Tue May 12 20:23:50 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data
../crop_patch.py ../../../sliver07/list.txt data/train.pkl \
    --patch_per_img 5000
