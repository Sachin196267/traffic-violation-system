#!/bin/bash
# Number-plate reader pipeline that runs unattended. Safe to re-run: finished steps are skipped.
#   1. fine-tune the plate reader (fast-plate-ocr cct-xs-v2 -> Indian plates); restarts from the best
#      checkpoint if the process dies (reboot, crash) - up to 3 times
#   2. export the best checkpoint to ONNX -> plates/ocr/indian_reader.onnx
#   3. evaluate on the held-out test plates -> plates/logs/eval_reader_v2.txt
# Progress: plates/logs/autopilot.log ; keeps plates/.heartbeat fresh while working.
cd "/Users/sachinnair/Desktop/traffic violation system/plates" || exit 1
export KERAS_BACKEND=torch
export SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")
L=logs/autopilot.log
log() { echo "[$(date '+%F %T')] $*" >> "$L"; touch .heartbeat; }
log "autopilot started"

best() { ls -t ocr/runs_v2/*/best.keras 2>/dev/null | head -1; }
tries=0
while [ ! -f ocr/TRAINING_DONE_V2 ]; do
  if ! pgrep -f "fast-plate-ocr train" >/dev/null; then
    if [ -n "$(best)" ] && grep -aq "Epoch.*early stopping\|Restoring model weights\|Training finished" logs/train_ocr_v2*.log 2>/dev/null; then
      touch ocr/TRAINING_DONE_V2; break
    fi
    tries=$((tries+1))
    [ $tries -gt 4 ] && { log "training failed 3 times - giving up (see logs/train_ocr_v2.log)"; exit 1; }
    start=ocr/runs/2026-09-27_16-44-03/best.keras
    [ -n "$(best)" ] && start=$(best)
    [ -f logs/train_ocr_v2.log ] && mv logs/train_ocr_v2.log logs/train_ocr_v2.try$tries.log
    log "starting training (try $tries) from $start"
    (cd ocr && nohup caffeinate -i fast-plate-ocr train --model-config-file cct_xs_v2_global_model_config.yaml \
        --plate-config-file plate_config.yaml --annotations train_aug.csv --val-annotations val.csv \
        --weights-path "$(cd .. && realpath "$start")" --epochs 120 --batch-size 32 --lr 0.0003 \
        --early-stopping-patience 25 --output-dir runs_v2 --region-loss-weight 0.0 --workers 2 \
        > ../logs/train_ocr_v2.log 2>&1; echo "Training finished" >> ../logs/train_ocr_v2.log) &
    sleep 60
  fi
  touch .heartbeat
  sleep 120
done
log "training done; best checkpoint $(best)"

if [ ! -f ocr/indian_reader_v2.onnx ]; then
  B=$(best); D=$(dirname "$B")
  # ONNX export only works with the TensorFlow Keras backend (torch backend fails on dynamic shapes)
  rm -rf ocr/export; mkdir -p ocr/export
  KERAS_BACKEND=tensorflow fast-plate-ocr export --model "$B" --plate-config-file ocr/plate_config.yaml \
      --format onnx --save-dir ocr/export >> logs/export.log 2>&1
  O=$(ls -t ocr/export/*.onnx 2>/dev/null | head -1)
  if [ -n "$O" ]; then cp "$O" ocr/indian_reader_v2.onnx; log "exported -> ocr/indian_reader_v2.onnx"
  else log "EXPORT FAILED - see logs/export.log"; exit 1; fi
fi

python3 eval_reader.py --onnx ocr/indian_reader_v2.onnx --config ocr/plate_config.yaml > logs/eval_reader_v2.txt 2>&1
log "held-out test: $(grep -a 'exact' logs/eval_reader_v2.txt | head -1)"
echo "AUTOPILOT DONE - remaining (manual): video pipeline test + REPORT.md" >> "$L"
log "autopilot finished"
