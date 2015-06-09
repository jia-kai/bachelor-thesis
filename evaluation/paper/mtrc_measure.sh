#!/bin/bash -e
# $File: mtrc_measure.sh
# $Date: Mon Jun 08 22:48:25 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data/mtrc/measure

cnt=0
for i in 800 5000 10000
do
    ../plot_roc.py -o data/mtrc/measure/${cnt}-0.pdf \
        ../data/roc/mtrc-d${i}-{cos,l2}.txt
    ../plot_roc.py -o data/mtrc/measure/${cnt}-1.pdf \
        ../data/roc/mtrc-mg-d${i}-{cos,l2}.txt
    cnt=$(($cnt+1))
done
