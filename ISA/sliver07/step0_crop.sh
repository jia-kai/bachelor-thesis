#!/bin/bash -e
# $File: step0_crop.sh
# $Date: Sun May 10 21:22:27 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data/l0
../crop_patch_l0.py ../../../sliver07/list.txt data/l0/train.pkl \
    --patch_per_img 50000
