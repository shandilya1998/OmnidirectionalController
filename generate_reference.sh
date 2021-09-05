#!/bin/sh

LOGDIR="assets/out/results_v2"
CLASS="simulations:Quadruped"
VERSION=1

python3 generate_reference.py --log_dir $LOGDIR --env_class $CLASS --version $VERSION
rm "$LOGDIR.zip"
zip -r "$LOGDIR.zip" "$LOGDIR/"
