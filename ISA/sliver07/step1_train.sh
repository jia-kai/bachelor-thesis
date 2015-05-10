#!/bin/bash
# $File: step1_train.sh
# $Date: Sun May 10 21:27:44 2015 +0800
# $Author: jiakai <jia.kai66@gmail.com>

export NASMIA_LOG_FILE=data/l0/train_log.txt
../train.py data/l0/train.pkl data/l0/model.pkl \
    --learning_rate 2 --out_dim 300
