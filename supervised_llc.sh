#!/bin/sh

EXPERIMENT=1
DATAPATH="assets/out/results_v3"

python3 supervised_llc.py --experiment $EXPERIMENT --datapath $DATAPATH
rm "$LOGDIR.zip"
zip -r "$LOGDIR.zip" "$LOGDIR/"
