#!/bin/bash -e
# $File: clsfy_datasize.sh
# $Date: Mon Jun 08 22:49:15 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data/clsfy

../plot_roc.py ../data/roc/clsfy-d{200,400,800,2000,5000}-l2.txt \
    -o data/clsfy/datasize.pdf
