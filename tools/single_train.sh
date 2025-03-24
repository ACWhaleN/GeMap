#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=$3  # 不再需要PORT参数

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 $(dirname "$0")/single_train.py $CONFIG --launcher none --gpus $GPUS ${@:4} --deterministic