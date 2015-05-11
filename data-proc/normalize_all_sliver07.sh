#!/bin/bash
# $File: normalize_all_sliver07.sh
# $Date: Mon May 11 21:17:41 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

cropper=$(realpath affine_normalize_img.py)
border_dist=$(realpath border_dist_tools/calc_border_dist.py)

if ! cd ../../sliver07/train-orig
then
    echo "training data do not exist"
    exit -1
fi

rm -rf ../train-cropped
rm -rf ../border-dist

mkdir ../train-cropped
mkdir ../border-dist

for i in $(seq 1 20)
do
    echo $i
    outnum=$(printf '%02d' $(($i-1)))
    output=../train-cropped/$outnum
    $cropper -i $(printf 'liver-orig%03d.nii.gz' $i) \
        --mask $(printf 'liver-seg%03d.nii.gz' $i) \
        -o $output --scale 0.8 --min_mask_border 11
    $border_dist ${output}-mask.nii.gz ../border-dist/${outnum}.nii.gz
done
