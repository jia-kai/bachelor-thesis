#!/bin/bash -e
# $File: mtrc_datasize.sh
# $Date: Tue Jun 09 10:29:34 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data/mtrc/datasize

cnt=0
for i in '' 'mg-'
do
    ../plot_roc.py -o data/mtrc/datasize/${cnt}.pdf \
        ../data/roc/mtrc-${i}d{800,5000,10000}-cos.txt
    cnt=$(($cnt+1))
done
