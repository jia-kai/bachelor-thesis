#!/bin/bash -e
# $File: step1_pairwise_match.sh
# $Date: Sun May 31 22:48:04 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

TRAIN_LIST=../../sliver07/list.txt
TEST_LIST=../../sliver07/list-test.txt
BORDER_DIST_DIR=../../sliver07/border-dist
OUTPUT_DIR=data/pairwise-match
FEATURE_DIR=data/feature

model_name=$1
measure=$2
shift 2 || true

if [ -z "$measure" ]
then
    echo "usage: $0 <model name> <dist measure> [-c]"
    exit -1
fi

output_dir=$OUTPUT_DIR/${model_name}-${measure}
feature_dir=$FEATURE_DIR/$model_name

if [ ! -d "$feature_dir" ]
then
    echo "feature dir $feature_dir does not exist; "\
        "please extract feature first"
    exit 2
fi

[ "$1" == "-c" ] || rm -rf $output_dir

if [ ! -d "$output_dir" ]
then
    mkdir -pv $output_dir
    echo "output dir: $output_dir"
    (echo "cmdline: $@"; date; env) > $output_dir/runtime_env
fi

for ftrain in $(cat $TRAIN_LIST)
do
    ftrain_name=$(basename $ftrain | cut -d. -f1)
    ftr_train=$feature_dir/${ftrain_name}.pkl
    bd_train=$BORDER_DIST_DIR/$(basename $ftrain)
    for ftest in $(cat $TEST_LIST)
    do
        ftest_name=$(basename $ftest | cut -d. -f1)
        ftr_test=$feature_dir/${ftest_name}.pkl
        echo $ftrain_name $ftest_name

        output=$output_dir/${ftrain_name}-${ftest_name}.pkl

        if [ -f $output ]
        then
            echo "$output already exists; skip"
            continue
        fi

        ./get_match.py \
            -r $ftr_train -b $bd_train -t $ftr_test \
            -o $output --measure $measure \
            --batch_size 4000 --nr_sample 3000

    done
done
