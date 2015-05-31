#!/bin/bash -e
# $File: step2_get_roc.sh
# $Date: Sun May 31 16:40:20 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

model_name=$1

if [ -z "$model_name" ]
then
    echo "usage: $0 <model name with dist measure>"
    exit -1
fi

mkdir -pv data/roc

./get_roc.py data/pairwise-match/$model_name/*.pkl \
    -o data/roc/${model_name}.txt
