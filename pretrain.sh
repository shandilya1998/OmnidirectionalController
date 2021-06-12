#!/bin/sh

OUT_PATH="assets/out/models"
EXPERIMENT=3
ENV="quadruped"
ENV_VERSION=2
ENV_CLASS="simulations:QuadrupedV2"
MODEL_CLASS="utils:Controller"

python3 pretrain.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --env $ENV \
    --env_version $ENV_VERSION \
    --env_class $ENV_CLASS \
    --model_class $MODEL_CLASS
