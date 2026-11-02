#!/bin/bash
# Retry downloading 10-min fixed-camera junction videos (YouTube bot block), then analyse them. Status: red_light/AUTOPILOT4_STATUS.md
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; ST=red_light/AUTOPILOT4_STATUS.md; D=datasets/signal_long
note() { echo "- $(date '+%m-%d %H:%M') $*" >> "$ST"; }
[ -f "$ST" ] || echo "# Autopilot 4: long junction videos" > "$ST"
note "started"
for round in $(seq 1 24); do
  left=0
  for p in "long-01 lWstIjx6080" "long-02 y6IRcg5vYzQ" "long-03 wqctLW0Hb_0"; do
    set -- $p
    [ -s "$D/$1.mp4" ] && continue
    .venv/bin/yt-dlp -q --no-warnings --js-runtimes node --cookies-from-browser chrome \
      -f "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]" --download-sections "*0-600" \
      --force-keyframes-at-cuts -o "$D/$1.%(ext)s" --remux-video mp4 "https://www.youtube.com/watch?v=$2" > /dev/null 2>&1
    if [ -s "$D/$1.mp4" ]; then
      note "downloaded $1"
      $PY red_light/observe.py "$D/$1.mp4" > "red_light/logs/observe_$1.log" 2>&1 && \
      $PY red_light/decide.py "$D/$1.mp4" --render > "red_light/logs/decide_$1.log" 2>&1 && \
      note "$1 analysed (automatic mode): $(grep -E '^\{' red_light/logs/decide_$1.log | tail -1)" || note "$1 analysis FAILED"
    else left=$((left+1)); fi
  done
  [ $left -eq 0 ] && break
  note "round $round: $left still blocked by YouTube - retry in 30 min"
  sleep 1800
done
note "AUTOPILOT4 DONE - remaining (manual): camera setup + grading of long-* clips"
