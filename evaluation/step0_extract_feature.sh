#!/bin/bash -e
# $File: step0_extract_feature.sh
# $Date: Sun May 17 11:44:13 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

FILE_LIST=../../sliver07/list-all.txt
OUTPUT_DIR=data/feature

model_runner=$1
model_name=$2

if [ ! -f "$FILE_LIST" ]
then
    echo "file list not found!"
    exit -1
fi

if [ -z "$model_name" ]
then
    echo "usage: $0 <model runner> <model name>"
    exit -1
fi


output_dir=$OUTPUT_DIR/$model_name
rm -rf $output_dir
mkdir -pv $output_dir
(echo "cmdline: $@"; date; env) > $output_dir/runtime_env

for i in $(cat $FILE_LIST)
do
    echo $i
    $model_runner $(dirname $FILE_LIST)/$i \
        $output_dir/$(basename $i | cut -d. -f1).pkl
done
