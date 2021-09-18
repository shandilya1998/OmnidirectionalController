#!/bin/sh

LOGDIR="assets/out/results_v9"
CLASS="simulations:Quadruped"
VERSION=2

python3 generate_reference.py --log_dir $LOGDIR --env_class $CLASS --version $VERSION
rm "$LOGDIR.zip"
zip -r "$LOGDIR.zip" "$LOGDIR/"
