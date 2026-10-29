#!/bin/bash
# More test data for the red-light + emergency-vehicle systems, processed unattended.
# Status: red_light/AUTOPILOT3_STATUS.md. Safe to re-run (skips finished work).
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python; Y=.venv/bin/yt-dlp; ST=red_light/AUTOPILOT3_STATUS.md; mkdir -p red_light/logs
note() { echo "- $(date '+%m-%d %H:%M') $*" | tee -a "$ST"; }
[ -f "$ST" ] || echo "# Autopilot 3: new junction + police/fire clips" > "$ST"
note "started"
dl() {   # dl <dir> <list>
  while read -r tag id; do
    [ -s "$1/$tag.mp4" ] && continue
    for try in 1 2; do
      "$Y" -q --no-warnings -f "bv*[height<=1080][ext=mp4]/bv*[height<=1080]" --download-sections "*0-240" \
        --force-keyframes-at-cuts -o "$1/$tag.%(ext)s" --remux-video mp4 "https://www.youtube.com/watch?v=$id" && break
      sleep 10
    done
    [ -s "$1/$tag.mp4" ] && note "downloaded $tag" || note "download FAILED $tag ($id)"
  done < "$2"
}
dl datasets/signal_videos datasets/signal_videos/list2.txt
dl datasets/emergency_videos datasets/emergency_videos/list.txt

# emergency clips: vehicle crops + CLIP scores, then review sheets for labelling
E=(); for f in datasets/emergency_videos/*.mp4; do E+=("$f"); done
$PY ambulance_training/extract_crops.py "${E[@]}" > red_light/logs/extract_emergency.log 2>&1 && note "emergency clips: crops extracted"
V=(); for f in datasets/emergency_videos/*.mp4; do V+=("$(basename "$f" .mp4)"); done
$PY ambulance_training/sheets.py videos 0.05 "${V[@]}" > red_light/logs/emergency_sheets.txt 2>&1 && note "label sheets ready: $(grep -c jpg red_light/logs/emergency_sheets.txt) (ambulance_training/sheets/vids_*.jpg - NOTE: overwrote old vids_* sheets, labels already saved)"

# junction clips: observe + automatic red-light decisions + videos
for f in datasets/signal_videos/sig-3[0-9].mp4; do
  n=$(basename "$f" .mp4)
  [ -f "red_light/runs/$n/obs.json" ] || $PY red_light/observe.py "$f" --max-sec 240 > "red_light/logs/observe_$n.log" 2>&1
  if $PY red_light/decide.py "$f" --render > "red_light/logs/decide_$n.log" 2>&1; then
    note "red light $n: $(grep -E '^\{' red_light/logs/decide_$n.log | tail -1)"
  else note "red light $n: FAILED $(tail -1 red_light/logs/decide_$n.log)"; fi
done
note "AUTOPILOT3 DONE - remaining (manual): label police/fire sheets, retrain probe, camera configs + grading for new junctions, report"
