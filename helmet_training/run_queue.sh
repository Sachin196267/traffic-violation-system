#!/bin/zsh
# Runs the controlled experiments sequentially. Logs in runs/helmet_nohelmet_final/logs/
cd "$(dirname "$0")/.."
mkdir -p runs/helmet_nohelmet_final/logs
for e in "$@"; do
  PYTHONPATH=helmet_training/_pydeps PYTHONWARNINGS=ignore .venv/bin/python helmet_training/train.py --exp $e > runs/helmet_nohelmet_final/logs/$e.log 2>&1
  echo "$e exit=$?" >> runs/helmet_nohelmet_final/logs/queue.log
done
