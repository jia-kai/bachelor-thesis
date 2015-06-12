#!/bin/bash -e
# $File: step0_crop_tiny.sh
# $Date: Sat May 23 12:13:32 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data
../crop_patch.py ../../../sliver07/list.txt data/train-400perimg.pkl \
    --patch_per_img 400
