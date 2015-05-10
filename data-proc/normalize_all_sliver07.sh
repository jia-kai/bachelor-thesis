#!/bin/bash
# $File: normalize_all_sliver07.sh
# $Date: Sun May 10 20:59:29 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

cropper=$(realpath affine_normalize_img.py)
cd ../../sliver07/train
rm -f ../train-cropped/*

for i in $(seq 1 20)
do
    echo $i
    $cropper -i $(printf 'liver-orig%03d.nii.gz' $i) \
        --mask $(printf 'liver-seg%03d.nii.gz' $i) \
        -o ../train-cropped/$(printf '%02d' $(($i-1))) \
        --scale 0.8
done
