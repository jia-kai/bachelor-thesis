#!/bin/bash -e
# $File: step0_crop_200.sh
# $Date: Mon Jun 08 09:29:27 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data
../crop_patch.py ../../../sliver07/list.txt data/train-200perimg.pkl \
    --patch_per_img 200
