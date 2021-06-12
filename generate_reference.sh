#!/bin/sh

LOGDIR="assets/out/reference"
CLASS="simulation:Quadruped"

python3 generate_reference.py --log_dir $LOGDIR --env_class $CLASS
rm "$LOGDIR.zip"
zip -r "$LOGDIR.zip" "$LOGDIR/"
