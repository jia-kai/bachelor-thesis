#!/bin/bash
# $File: normalize_all_sliver07.sh
# $Date: Mon May 11 19:56:20 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

cropper=$(realpath affine_normalize_img.py)
border_dist=$(realpath border_dist_tools/calc_border_dist.py)

set +e
cd ../../sliver07/train-orig
rm -rf ../train-cropped
rm -rf ../border-dist

mkdir ../train-cropped
mkdir ../border-dist

set -e

for i in $(seq 1 20)
do
    echo $i
    outnum=$(printf '%02d' $(($i-1)))
    output=../train-cropped/$outnum
    $cropper -i $(printf 'liver-orig%03d.nii.gz' $i) \
        --mask $(printf 'liver-seg%03d.nii.gz' $i) \
        -o $output --scale 0.8
    $border_dist ${output}-mask.nii.gz ../border-dist/${outnum}.nii.gz
done
