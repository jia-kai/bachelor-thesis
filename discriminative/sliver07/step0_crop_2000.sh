#!/bin/bash -e
# $File: step0_crop_2000.sh
# $Date: Mon Jun 01 22:48:15 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data
../crop_patch.py ../../../sliver07/list.txt data/train-d2000.pkl \
    --patch_per_img 2000
