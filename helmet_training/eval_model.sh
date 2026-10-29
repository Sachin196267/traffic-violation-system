#!/bin/zsh
# Usage: helmet_training/eval_model.sh <tag> <weights> <imgsz> [device]
# Runs the full image-level evaluation + PH rider-level OOD test for one model.
cd "$(dirname "$0")/.."
TAG=$1; W=$2; SZ=$3; DEV=${4:-cpu}
.venv/bin/python helmet_training/evaluate.py --weights "$W" --imgsz $SZ --tag $TAG --device $DEV > helmet_training/reports/eval_$TAG.log 2>&1 || { echo "EVAL FAIL $TAG"; tail -5 helmet_training/reports/eval_$TAG.log; exit 1; }
C=$(.venv/bin/python -c "import json;print(','.join(map(str,json.load(open('helmet_training/reports/eval/$TAG/report.json'))['operating_conf(chosen_on_val)'])))")
.venv/bin/python helmet_training/eval_ph_rider_level.py --weights "$W" --imgsz $SZ --conf $C --tag $TAG --device $DEV >> helmet_training/reports/eval_$TAG.log 2>&1 || { echo "PH FAIL $TAG"; exit 1; }
echo "done $TAG conf=$C"
