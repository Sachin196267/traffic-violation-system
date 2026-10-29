#!/bin/bash
# observe.py on every usable signal clip (skips talk / slide videos); resumable.
cd "$(dirname "$0")/.." || exit 1
for f in datasets/signal_videos/sig-*.mp4; do
  n=$(basename "$f" .mp4)
  case "$n" in sig-17|sig-18|sig-19|sig-20|sig-24) continue;; esac
  [ -f "red_light/runs/$n/obs.json" ] && [ "$n" != "sig-06" ] && continue
  echo "== $n $(date +%T)"
  .venv/bin/python red_light/observe.py "$f" --max-sec 240 2>&1 | grep -E "pre-pass|done|Error|Traceback" 
done
echo "OBSERVE ALL DONE"
