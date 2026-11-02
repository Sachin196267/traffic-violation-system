#!/bin/bash
# Runs the rest of the triple-rider v3 pipeline unattended.
#   1. watchdog: if training dies before "final weights", restart only the folds not yet done
#   2. thresholds from held-out predictions (0 wrong TRIPLE calls, not-sure <= 1/12)
#   3. end-to-end video tests, each video with the fold model that never saw it
# Progress: triple_training/logs/autopilot.log ; marks steps in PROGRESS.md.
cd "/Users/sachinnair/Desktop/traffic violation system" || exit 1
T=triple_training; L=$T/logs/autopilot.log
log() { echo "[$(date '+%F %T')] $*" >> "$L"; touch $T/.heartbeat; }
log "autopilot started"

tries=0
while ! grep -aq "final weights" $T/logs/train_cv_v3.log 2>/dev/null; do
  if ! pgrep -f "train_cv.py" >/dev/null; then
    tries=$((tries+1))
    [ $tries -gt 3 ] && { log "training failed 3 times - giving up, see train_cv_v3.log"; exit 1; }
    done_folds=$(grep -ao "fold [0-4]: .*held-out" $T/logs/train_cv_v3*.log 2>/dev/null | grep -o "fold [0-4]" | awk '{print $2}' | sort -u)
    todo=""; for k in 0 1 2 3 4; do echo "$done_folds" | grep -qx "$k" || todo="$todo $k"; done
    log "training not running (try $tries) - restarting folds:[$todo] + final"
    cp $T/logs/train_cv_v3.log $T/logs/train_cv_v3.try$tries.log 2>/dev/null
    if [ -n "$todo" ]; then
      nohup sh -c "python3 $T/train_cv.py --folds$todo --epochs 15 --imgsz 256 --resume && python3 $T/train_cv.py --final --epochs 15 --imgsz 256" > $T/logs/train_cv_v3.log 2>&1 &
    else
      nohup python3 $T/train_cv.py --final --epochs 15 --imgsz 256 > $T/logs/train_cv_v3.log 2>&1 &
    fi
    sleep 30
    nohup caffeinate -i -w $(pgrep -f "train_cv.py" | head -1) >/dev/null 2>&1 &
  fi
  touch $T/.heartbeat
  sleep 120
done
log "training finished"
sed -i '' 's/^- \[ \] 3 /- [x] 3 /' $T/PROGRESS.md

python3 $T/combine.py --max-not-sure 0.0833 > $T/logs/eval_v3.log 2>&1
log "thresholds: $(grep chosen $T/logs/eval_v3.log)"

V=helmet_training/robust/vids; D=datasets/triple_videos; O=$T/video_out_v3; W=runs/triple_rider
fold_of() { python3 -c "import json,sys;S=json.load(open('$T/samples.json'));print(next((s['fold'] for s in S if s['video']==sys.argv[1]),-1))" "$1"; }
for v in $V/indore.mp4 $V/pune.mp4 $V/jaipur.mp4 $V/mum_traffic.mp4 $D/tpt-26.mp4 $D/tpt-15.mp4 $D/hydpol-04.mp4; do
  name=$(basename "$v" .mp4); f=$(fold_of "$name")
  [ "$f" -lt 0 ] 2>/dev/null && continue
  log "video test $name with fold$f model"
  python3 $T/run_video.py "$v" --max-sec 90 --weights $W/fold$f/weights/last.pt --out $O >> $T/logs/video_tests_v3.log 2>&1
  touch $T/.heartbeat
done
log "video tests done: $(grep -a "{'triple'" $T/logs/video_tests_v3.log | tr '\n' ' ')"
echo "AUTOPILOT DONE - remaining (manual): REPORT.md (step 6)" >> "$L"
log "autopilot finished"
