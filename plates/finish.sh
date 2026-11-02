#!/bin/bash
# After autopilot (v2 reader) finishes: choose READ threshold, then run the 4 test videos. Survives session limits.
cd "/Users/sachinnair/Desktop/traffic violation system" || exit 1
L=plates/logs/autopilot.log; log() { echo "[$(date '+%F %T')] $*" >> "$L"; touch plates/.heartbeat; }
until grep -q "exported -> ocr/indian_reader_v2.onnx" "$L" && grep -q "held-out test" <(tail -3 "$L"); do
  touch plates/.heartbeat; sleep 60
done
python3 plates/tune.py --target 0.99 > plates/logs/tune_v2.log 2>&1; log "v2 threshold: $(grep chosen plates/logs/tune_v2.log)"
O=plates/video_out_v2; D=datasets/triple_videos; V=helmet_training/robust/vids
for a in "$D/tpt-26.mp4" "$D/tpt-23.mp4" "$V/mum_traffic.mp4" "$V/pune.mp4"; do
  python3 plates/run_video.py "$a" --max-sec 60 --out $O >> plates/logs/video_v2.log 2>&1 &
done
while pgrep -f "plates/run_video.py" >/dev/null; do touch plates/.heartbeat; sleep 60; done
log "v2 video tests: $(grep -h "{" plates/logs/video_v2.log | tr '\n' ' ')"
echo "FINISH DONE - remaining (manual): check READ plates by eye + REPORT.md" >> "$L"
