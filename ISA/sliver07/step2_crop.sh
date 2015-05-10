#!/bin/bash -e
# $File: step1_crop.sh
# $Date: Sun May 10 21:35:52 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data/l1
../crop_patch_l1.py data/l0/model.pkl ../../../sliver07/list.txt \
    data/l1/train.pkl \
    --patch_per_img 50000
