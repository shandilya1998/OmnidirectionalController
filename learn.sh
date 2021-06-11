#!/bin/sh

OUT_PATH="assets/out/models"
EXPERIMENT=2
ENV="quadruped"
ENV_VERSION=2
ENV_CLASS="simulations:QuadrupedV2"

python3 train.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --env $ENV \
    --env_version $ENV_VERSION \
    --env_class $ENV_CLASS \
    --ppo \
    --render
