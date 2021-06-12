#!/bin/sh

OUT_PATH="assets/out/models"
EXPERIMENT=1
MODEL_DIR="assets/out/models/exp${EXPERIMENT}"
ENV="quadruped"
ENV_VERSION=1
ENV_CLASS="simulations:Quadruped"

python3 evaluate.py \
    --experiment $EXPERIMENT \
    --out_path $OUT_PATH \
    --env $ENV \
    --env_version $ENV_VERSION \
    --env_class $ENV_CLASS \
    --model_dir $MODEL_DIR \
    --ppo
