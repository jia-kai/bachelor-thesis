#!/bin/bash
# $File: run.sh
# $Date: Sun May 10 22:19:13 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

fdir=$(dirname $0)
$fdir/../apply_model_on_data.py \
    --l0 $fdir/data/l0/model.pkl \
    --l1 $fdir/data/l1/model.pkl \
    $@
