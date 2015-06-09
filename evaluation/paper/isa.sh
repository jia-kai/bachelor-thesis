#!/bin/bash
# $File: isa.sh
# $Date: Tue Jun 09 11:03:39 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

mkdir -pv data
../plot_roc.py ../data/roc/ISA-{cos,l2}.txt -o data/isa.pdf
