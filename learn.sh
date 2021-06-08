#!/bin/sh

OUT_PATH="assets/out/models"
EXPERIMENT=1
ENV="quadruped"
ENV_VERSION=1
ENV_CLASS="simulations.quadruped:Quadruped"

python3 train.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --env $ENV \
    --env_version $ENV_VERSION \
    --env_class $ENV_CLASS \
    --ppo
