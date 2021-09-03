#!/bin/sh

LOGDIR="assets/out/results"
CLASS="simulations:Quadruped"
VERSION=0

python3 generate_reference.py --log_dir $LOGDIR --env_class $CLASS --version $VERSION
rm "$LOGDIR.zip"
zip -r "$LOGDIR.zip" "$LOGDIR/"
