#!/bin/bash
# Scan the downloaded signal clips (skipping talk/slide videos) with the IDD and COCO detectors.
cd "$(dirname "$0")/.." || exit 1
V=()
for f in datasets/signal_videos/sig-*.mp4; do
  case "$(basename "$f" .mp4)" in sig-17|sig-18|sig-19|sig-20|sig-24) continue;; esac
  V+=("$f")
done
.venv/bin/python signal_training/scan_lights.py --every 1 --max-frames 60 "${V[@]}"
.venv/bin/python signal_training/scan_lights.py --every 1 --max-frames 60 --weights yolo11s.pt --tag _coco "${V[@]}"
echo "SCAN DONE"
