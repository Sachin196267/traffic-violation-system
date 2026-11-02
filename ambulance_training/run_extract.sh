#!/bin/bash
# Crops from the ambulance clips, then negatives-rich Indian traffic videos (white vans etc.).
cd "$(dirname "$0")/.." || exit 1
A=(datasets/ambulance_videos/amb-*.mp4)
N=(datasets/helmet_videos/{chennai,indore,jaipur,lucknow,pune,kol_signal,mum_traffic,traffic_sound,vns_cross,ahd}.mp4 datasets/triple_videos/{hydpol-05,tpt-25,vjw-01}.mp4)
.venv/bin/python ambulance_training/extract_crops.py "${A[@]}"
.venv/bin/python ambulance_training/extract_crops.py --per-track 3 --max-sec 120 "${N[@]}"
echo "EXTRACT DONE"
