#!/bin/bash -e
# $File: mtrc_pca.sh
# $Date: Fri Jun 12 10:35:03 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data/mtrc/pca

cnt=0
for i in d10000 mg-d10000 d5000 mg-d5000
do
    ../plot_roc.py -o data/mtrc/pca/${cnt}.pdf \
        ../data/roc/mtrc-${i}-{cos,pca*-l2,pca*-cos}.txt
    cnt=$(($cnt+1))
done
