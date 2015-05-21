#!/bin/bash -e
# $File: step0_crop_small.sh
# $Date: Fri May 15 23:57:26 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data
../crop_patch.py ../../../sliver07/list.txt data/train-800perimg.pkl \
    --patch_per_img 800
