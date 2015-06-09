#!/bin/bash -e
# $File: clsfy_pca.sh
# $Date: Tue Jun 09 11:22:02 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data/clsfy/pca

cnt=0
for i in 400 800
do
    ../plot_roc.py -o data/clsfy/pca/${cnt}.pdf \
        ../data/roc/clsfy-d${i}-{l2,pca-l2,pca-cos}.txt
    cnt=$(($cnt+1))
done
