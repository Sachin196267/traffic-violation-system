#!/bin/bash
# Testing steps that run unattended. Safe to re-run. Status: echallan/AUTOPILOT2_STATUS.md
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; ST=echallan/AUTOPILOT2_STATUS.md; mkdir -p echallan/logs
note() { echo "- $(date '+%m-%d %H:%M') $*" | tee -a "$ST"; }
[ -f "$ST" ] || echo "# e-challan / red-light test autopilot" > "$ST"
note "started"

# 1. Indian fine-tune of the light model (restart once if it died)
for try in 1 2; do
  while pgrep -f "signal_training/finetune.py" > /dev/null; do sleep 60; done
  grep -q "FINETUNE DONE" signal_training/logs/finetune.log && break
  note "fine-tune stopped early (try $try) - restarting (finished models are kept)"
  nohup caffeinate -i $PY signal_training/finetune.py >> signal_training/logs/finetune.log 2>&1 &
  sleep 60
done
FT=runs/signal_state_ft/final/weights/last.pt
if [ -f "$FT" ]; then
  mkdir -p runs/signal_state/weights; cp runs/signal_state/weights/last.pt runs/signal_state/weights/lisa_only.pt 2>/dev/null
  cp "$FT" runs/signal_state/weights/last.pt
  note "Indian fine-tuned light model installed (LISA-only kept as lisa_only.pt); held-out clips: $(tr -d '\n' < runs/signal_state_ft/report.json | cut -c1-600)"
fi

# 2. wait for the observe pass, then red-light decisions + videos on every signal clip with the new models
while pgrep -f "red_light/observe_all.sh" > /dev/null; do sleep 60; done
for d in red_light/runs/sig-*/; do
  n=$(basename "$d"); [ -f "$d/obs.json" ] || continue
  if $PY red_light/decide.py "datasets/signal_videos/$n.mp4" --render > "echallan/logs/decide_$n.log" 2>&1; then
    note "red light $n: $(grep -E '^\{' echallan/logs/decide_$n.log | tail -1)"
  else note "red light $n: FAILED $(tail -1 echallan/logs/decide_$n.log)"; fi
done

# 3. full e-challan runs on test clips (helmet + triple + red light; speed needs a calibration)
for v in datasets/signal_videos/sig-03.mp4 datasets/signal_videos/sig-06.mp4 datasets/signal_videos/sig-21.mp4 \
         datasets/triple_videos/tpt-15.mp4 datasets/triple_videos/tpt-26.mp4 datasets/ambulance_videos/amb-14.mp4; do
  n=$(basename "$v" .mp4)
  if $PY echallan/run.py "$v" --max-sec 120 > "echallan/logs/echallan_$n.log" 2>&1; then
    note "e-challan $n: $(tail -1 echallan/logs/echallan_$n.log)"
  else note "e-challan $n: FAILED $(tail -2 echallan/logs/echallan_$n.log | tr '\n' ' ')"; fi
done
note "AUTOPILOT2 DONE - remaining (manual): grade red-light decisions + e-challan outputs, REPORT.md"
