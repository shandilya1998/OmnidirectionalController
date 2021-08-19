#!/bin/sh

LOGDIR="assets/out/results_v2"
CLASS="simulations:Quadruped"

python3 generate_reference.py --log_dir $LOGDIR --env_class $CLASS
rm "$LOGDIR.zip"
zip -r "$LOGDIR.zip" "$LOGDIR/"
