#!/bin/bash
# Runs the rest of the red-light / ambulance pipeline unattended. Safe to re-run.
# Status: red_light/AUTOPILOT_STATUS.md (short), details in red_light/logs/autopilot.log.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; K=.venv/bin/kaggle
mkdir -p red_light/logs; L=red_light/logs/autopilot.log; ST=red_light/AUTOPILOT_STATUS.md
note() { echo "- $(date '+%m-%d %H:%M') $*" | tee -a "$ST"; }
[ -f "$ST" ] || echo "# Red-light autopilot status" > "$ST"
note "autopilot started"

# 1. light-state model from Kaggle (poll up to 8 h; one automatic re-push if it errors)
W=runs/signal_state/weights/last.pt
pushed=0
while [ ! -f "$W" ]; do
  s=$($K kernels status sachin196267/tvs-tl-state 2>&1)
  echo "$(date +%T) $s" >> "$L"
  if echo "$s" | grep -q COMPLETE; then
    rm -rf runs/signal_state_kaggle; $K kernels output sachin196267/tvs-tl-state -p runs/signal_state_kaggle >> "$L" 2>&1
    if [ -f runs/signal_state_kaggle/out/run/weights/last.pt ]; then
      mkdir -p runs/signal_state/weights; cp runs/signal_state_kaggle/out/run/weights/last.pt "$W"
      cp runs/signal_state_kaggle/out/report.json runs/signal_state/ 2>/dev/null
      note "light-state model downloaded -> $W (held-out LISA/TVD confusion in runs/signal_state/report.json)"
    else note "Kaggle tl-state finished but no weights - see runs/signal_state_kaggle/*.log"; break; fi
  elif echo "$s" | grep -qE "ERROR|CANCEL"; then
    if [ $pushed -eq 0 ]; then pushed=1; note "Kaggle tl-state failed - pushing it once more"
      $K kernels push -p kaggle_jobs/tl_state_train >> "$L" 2>&1
    else note "Kaggle tl-state failed twice - stopping light-state step"; break; fi
  fi
  sleep 300
done

# 2. CLIP pre-sort of the web ambulance images (for labelling later)
for try in $(seq 1 60); do
  s=$($K kernels status sachin196267/tvs-amb-clip-cpu 2>&1)
  if echo "$s" | grep -q COMPLETE; then
    $K kernels output sachin196267/tvs-amb-clip-cpu -p ambulance_training/clip_web >> "$L" 2>&1
    note "CLIP web pre-sort downloaded -> ambulance_training/clip_web/clip_scores.csv"; break
  elif echo "$s" | grep -qE "ERROR|CANCEL"; then note "CLIP web pre-sort job failed (not needed for the video tests)"; break; fi
  sleep 300
done

# 3. wait for the local observe pass and crop extraction (started earlier with nohup); restart if they died
for job in "red_light/observe_all.sh:red_light/observe_all.log:OBSERVE ALL DONE" "ambulance_training/run_extract.sh:ambulance_training/logs/extract.log:EXTRACT DONE"; do
  IFS=: read -r script logf marker <<< "$job"
  for try in 1 2 3; do
    while pgrep -f "$script" > /dev/null; do sleep 120; done
    grep -q "$marker" "$logf" && break
    note "$script stopped before finishing (try $try) - restarting (it skips finished videos)"
    nohup caffeinate -i "$script" >> "$logf" 2>&1 &
    sleep 60
  done
  note "$script: $(grep -c . "$logf") log lines, finished: $(grep -q "$marker" "$logf" && echo yes || echo NO)"
done
$PY ambulance_training/summarize.py > red_light/logs/ambulance_summary.txt 2>&1 && note "ambulance summary (CLIP zero-shot, per video): red_light/logs/ambulance_summary.txt"

# 4. decisions + annotated videos for every observed clip
if [ -f "$W" ]; then
  for d in red_light/runs/*/; do
    n=$(basename "$d"); [ -f "$d/obs.json" ] || continue
    [ -f "$d/decisions.json" ] && [ "$d/decisions.json" -nt "$d/obs.json" ] && continue
    v=$(ls datasets/signal_videos/$n.mp4 2>/dev/null || ls datasets/*/$n.mp4 2>/dev/null | head -1)
    if $PY red_light/decide.py "$v" --render > "red_light/logs/decide_$n.log" 2>&1; then
      note "$n: $(grep -E '^\{' red_light/logs/decide_$n.log | tail -1)"
    else note "$n: decide.py FAILED - $(tail -1 red_light/logs/decide_$n.log)"; fi
  done
else note "no light-state model - decisions skipped"; fi
note "AUTOPILOT DONE - remaining (manual): label ambulances (sheets.py), train probe, grade clips, REPORT.md"
