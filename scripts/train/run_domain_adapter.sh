#!/usr/bin/env bash

GPUS="0,1,2,3"


bash scripts/train/FAE_train.sh \
    -v ${GPUS} \