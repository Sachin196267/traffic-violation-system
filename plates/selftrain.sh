#!/bin/bash
# Self-training round (runs unattended, safe to re-run): waits for finish.sh, then
#  1. reads plates on ~25 NON-test traffic videos, saving every plate crop (plates/dump, plates/video_self)
#  2. selftrain.py -> ocr/train_self.csv (confident vehicles label their own blurry crops)
#  3. retrains the reader (v3) from the v2 best, exports ocr/indian_reader_v3.onnx, re-tunes, re-runs the 4 test videos
cd "/Users/sachinnair/Desktop/traffic violation system" || exit 1
export SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")
L=plates/logs/autopilot.log; log() { echo "[$(date '+%F %T')] $*" >> "$L"; touch plates/.heartbeat; }
ME=$$; (while kill -0 $ME 2>/dev/null; do touch plates/.heartbeat; sleep 300; done) &   # heartbeat while alive
until grep -q "FINISH DONE" "$L"; do touch plates/.heartbeat; sleep 60; done
log "self-training started"
V=helmet_training/robust/vids; D=datasets/triple_videos
VIDS="$V/ahd.mp4 $V/blr_pov.mp4 $V/chennai.mp4 $V/jaipur.mp4 $V/lucknow.mp4 $V/kol_signal.mp4 $V/indore.mp4 $V/sdelhi.mp4 $V/mumbai.mp4 $V/rally.mp4 $V/hyd_night.mp4 $V/vns_cross.mp4"
for k in 01 02 04 05 06 08 09 10 11 12 13 15 17 18 20 22 25 27; do VIDS="$VIDS $D/tpt-$k.mp4"; done
run() { for v in "$@"; do s=$(basename "$v" .mp4); [ -f plates/video_self/${s}_plates.json ] && continue
          python3 plates/run_video.py "$v" --max-sec 120 --out plates/video_self --dump plates/dump >> plates/logs/selfrun.log 2>&1; done; }
set -- $VIDS; n=$#; a=(); b=(); c=(); i=0
for v in "$@"; do case $((i%3)) in 0) a+=("$v");; 1) b+=("$v");; 2) c+=("$v");; esac; i=$((i+1)); done
run "${a[@]}" & run "${b[@]}" & run "${c[@]}" &
while pgrep -f "plates/run_video.py" >/dev/null; do touch plates/.heartbeat; sleep 60; done
wait
log "self-training videos done: $(ls plates/video_self/*_plates.json | wc -l) videos"
python3 plates/selftrain.py > plates/logs/selftrain.log 2>&1; log "$(cat plates/logs/selftrain.log)"
cd plates/ocr
if [ ! -f runs_v3_done ]; then
  B=$(ls -t runs_v2/*/best.keras | head -1)
  KERAS_BACKEND=torch caffeinate -i fast-plate-ocr train --model-config-file cct_xs_v2_global_model_config.yaml \
    --plate-config-file plate_config.yaml --annotations train_self.csv --val-annotations val.csv \
    --weights-path "$(realpath "$B")" --epochs 80 --batch-size 32 --lr 0.0002 --early-stopping-patience 15 \
    --output-dir runs_v3 --region-loss-weight 0.0 --workers 2 > ../logs/train_ocr_v3.log 2>&1 && touch runs_v3_done
fi
B=$(ls -t runs_v3/*/best.keras | head -1); rm -rf export; mkdir -p export
KERAS_BACKEND=tensorflow fast-plate-ocr export --model "$B" --plate-config-file plate_config.yaml --format onnx --save-dir export >> ../logs/export.log 2>&1
cp export/*.onnx indian_reader_v3.onnx && cd .. && log "v3 exported"
python3 eval_reader.py --onnx ocr/indian_reader_v3.onnx --config ocr/plate_config.yaml > logs/eval_reader_v3.txt 2>&1
log "v3 held-out: $(grep -a exact logs/eval_reader_v3.txt | head -1)"
V2=$(grep -ao "exact [0-9.]*" logs/eval_reader_v2.txt | head -1 | awk '{print $2}')
V3=$(grep -ao "exact [0-9.]*" logs/eval_reader_v3.txt | head -1 | awk '{print $2}')
if python3 -c "import sys; sys.exit(0 if float('${V3:-0}') >= float('${V2:-1}') else 1)"; then
  log "v3 ($V3) >= v2 ($V2): switching to v3"
else
  log "v3 ($V3) WORSE than v2 ($V2): keeping v2, skipping v3 tests"
  echo "SELFTRAIN DONE - v3 rejected (worse than v2); remaining (manual): check v2 READ plates by eye + REPORT.md" >> "$L"
  exit 0
fi
cd .. && python3 - <<'PY'
p='plates/run_video.py'; s=open(p).read()
s=s.replace('"ocr/indian_reader_v2.onnx" if (P / "ocr/indian_reader_v2.onnx").exists()','"ocr/indian_reader_v3.onnx" if (P / "ocr/indian_reader_v3.onnx").exists()')
open(p,'w').write(s)
p='plates/tune.py'; s=open(p).read(); s=s.replace('indian_reader_v2','indian_reader_v3'); open(p,'w').write(s)
PY
python3 plates/tune.py --target 0.99 > plates/logs/tune_v3.log 2>&1; log "v3 threshold: $(grep chosen plates/logs/tune_v3.log)"
O=plates/video_out_v3
for a in "$D/tpt-26.mp4" "$D/tpt-23.mp4" "$V/mum_traffic.mp4" "$V/pune.mp4"; do
  python3 plates/run_video.py "$a" --max-sec 60 --out $O >> plates/logs/video_v3.log 2>&1 &
done
while pgrep -f "plates/run_video.py" >/dev/null; do touch plates/.heartbeat; sleep 60; done
log "v3 video tests: $(grep -h "{" plates/logs/video_v3.log | tr '\n' ' ')"
echo "SELFTRAIN DONE - remaining (manual): check READ plates by eye (v2 + v3) + REPORT.md" >> "$L"
