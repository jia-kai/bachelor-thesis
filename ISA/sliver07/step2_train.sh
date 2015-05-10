#!/bin/bash
# $File: step2_train.sh
# $Date: Sun May 10 21:55:15 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

export NASMIA_LOG_FILE=data/l1/train_log.txt
../train.py data/l1/train.pkl data/l1/model.pkl \
    --learning_rate 2 --out_dim 50 --subspace_size 4
