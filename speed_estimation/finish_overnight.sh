#!/bin/bash
# Runs the remaining automatable steps unattended; safe to re-run. Log + status in runs/overnight_*.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=speed_estimation/runs/overnight.log
ST=speed_estimation/runs/overnight_status.md
note() { echo "- $(date '+%H:%M') $*" | tee -a "$ST"; }
echo "# Overnight status ($(date '+%Y-%m-%d'))" > "$ST"

# 1. wait for the VS13 evaluation; restart it (resumable) if it died before finishing, max 3 tries
for try in 1 2 3; do
  while pgrep -f "vs13_eval.py" > /dev/null; do sleep 60; done
  if grep -q "^measured" speed_estimation/runs/vs13_eval.log 2>/dev/null; then note "VS13 evaluation finished"; break; fi
  note "VS13 evaluation stopped without finishing (try $try) - last lines: $(tail -2 speed_estimation/runs/vs13_eval.log | tr '\n' ' ')"
  for d in speed_estimation/runs/Peugeot208_*; do [ -f "$d/speeds.json" ] || rm -rf "$d"; done
  $PY speed_estimation/vs13_eval.py datasets/speed/vs13 --resume > speed_estimation/runs/vs13_eval.log 2>&1
done
note "VS13 summary: $(grep '^measured' speed_estimation/runs/vs13_eval.log)"

# 2. report section
if $PY speed_estimation/finish_report.py >> $LOG 2>&1; then note "REPORT.md: VS13 section added"; else note "REPORT.md update FAILED (see overnight.log)"; fi

# 3. demo videos with the final code
if $PY speed_estimation/speed.py "datasets/speed/m6/M6 Motorway Traffic.mp4" --calib speed_estimation/calibrations/m6_all_lines.json --tag _calibrated >> $LOG 2>&1; then
  note "M6 calibrated demo video re-rendered: speed_estimation/runs/M6 Motorway Traffic/M6 Motorway Traffic_speed_calibrated.mp4"
else note "M6 demo FAILED (see overnight.log)"; fi
if $PY speed_estimation/speed.py "videos/bombay traffic [zxxfvP8-lrU].mp4" >> $LOG 2>&1; then
  note "Bombay demo video re-rendered: speed_estimation/runs/bombay traffic [zxxfvP8-lrU]/bombay traffic [zxxfvP8-lrU]_speed.mp4"
else note "Bombay demo FAILED (see overnight.log)"; fi
note "all automatic steps done"
