#!/bin/bash -e
# $File: clsfy_measure.sh
# $Date: Wed Jun 10 00:26:01 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data/clsfy/measure

cnt=0
for i in 200 400 800 2000 5000
do
    ../plot_roc.py -o data/clsfy/measure/${cnt}.pdf \
        ../data/roc/clsfy-d${i}-{cos,l2}.txt
    cnt=$(($cnt+1))
done
