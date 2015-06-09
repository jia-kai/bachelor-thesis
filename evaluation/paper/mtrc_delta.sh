#!/bin/bash -e
# $File: mtrc_delta.sh
# $Date: Mon Jun 08 23:48:51 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data/mtrc/delta

cnt=0
for i in 800 5000 10000
do
    ../plot_roc.py -o data/mtrc/delta/${cnt}.pdf \
        ../data/roc/mtrc-{mg-,}d${i}-cos.txt
    cnt=$(($cnt+1))
done
