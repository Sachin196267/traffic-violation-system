#!/bin/bash
# After autopilot4 analyses a long clip: full e-challan run + grading sheets, so the next manual step is only grading.
cd "$(dirname "$0")/.." || exit 1
ST=red_light/AUTOPILOT4_STATUS.md
while ! grep -q "AUTOPILOT4 DONE" "$ST"; do
  for f in datasets/signal_long/long-0*.mp4; do
    [ -f "$f" ] || continue; n=$(basename "$f" .mp4)
    grep -q "$n analysed" "$ST" || continue
    [ -f "echallan/runs/$n/challans.json" ] && continue
    .venv/bin/python echallan/run.py "$f" --checks redlight,helmet --max-sec 600 > "echallan/logs/echallan_$n.log" 2>&1 \
      && echo "- $(date '+%m-%d %H:%M') $n e-challan: $(tail -1 echallan/logs/echallan_$n.log | cut -c1-200)" >> "$ST"
    .venv/bin/python echallan/grade_sheet.py "$n" --decision REVIEW > /dev/null 2>&1
  done
  sleep 600
done
