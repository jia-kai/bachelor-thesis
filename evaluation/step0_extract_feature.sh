#!/bin/bash -e
# $File: step0_extract_feature.sh
# $Date: Sun May 31 20:19:08 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

FILE_LIST=../../sliver07/list-all.txt
OUTPUT_DIR=data/feature

model_runner=$1
model_name=$2
shift 2 || true

if [ ! -f "$FILE_LIST" ]
then
    echo "file list not found!"
    exit -1
fi

if [ -z "$model_name" ]
then
    echo "usage: $0 <model runner> <model name> [-c]"
    exit -1
fi

output_dir=$OUTPUT_DIR/$model_name

[ "$1" == "-c" ] || rm -rf $output_dir

if [ ! -d "$output_dir" ]
then
    mkdir -pv $output_dir
    (echo "cmdline: $@"; date; env) > $output_dir/runtime_env
fi

for i in $(cat $FILE_LIST)
do
    echo $i
    cur_out=$output_dir/$(basename $i | cut -d. -f1).pkl
    if [ -f "$cur_out" ]
    then
        echo "$cur_out already exists, ignore"
        continue
    fi
    $model_runner $(dirname $FILE_LIST)/$i $cur_out

done
