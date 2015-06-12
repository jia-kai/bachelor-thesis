#!/bin/bash -e
# $File: border_dist_stat.sh
# $Date: Fri Jun 12 15:10:26 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data/border-dist-stat

../plot_border_dist_stat.py -o data/border-dist-stat/l2.pdf \
    ../data/border-dist-stat/{ISA-l2.txt,clsfy-d400-l2.txt}

../plot_border_dist_stat.py -o data/border-dist-stat/cos.pdf \
    ../data/border-dist-stat/{clsfy-d5000-cos.txt,mtrc-mg-d10000-pca-cos.txt}
