#!/bin/bash
# $File: all.sh
# $Date: Tue Jun 09 13:15:48 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

../plot_roc.py -o data/all.pdf --no_errbar \
    ../data/roc/{ISA-cos.txt,clsfy-d400-l2.txt,clsfy-d400-pca-cos.txt,mtrc-d5000-cos.txt,mtrc-mg-d10000-pca-cos.txt}
