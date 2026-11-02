#!/bin/bash
# Downloads the ambulance / emergency-vehicle datasets (and the BDD traffic-light crops) into datasets/external/.
# Safe to re-run: skips datasets already unzipped, verifies each one produced files.
cd "$(dirname "$0")/.." || exit 1
K=.venv/bin/kaggle
for d in ganeshmohane/indian-emergency-vehicles-dataset abhisheksinghblr/emergency-vehicles-identification \
         daudshah/pakistani-ambulances-dataset ikramali585/emergency-vs-non-emergency-vehicles \
         guisahanes/traffic-violation-detection-dataset; do
  out="datasets/external/${d/\//_}"
  if [ -f "$out/.done" ]; then echo "skip $d"; continue; fi
  for try in 1 2 3; do
    $K datasets download "$d" -p "$out" --unzip -q && break
    echo "retry $try $d"; sleep 20
  done
  n=$(find "$out" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
  if [ "$n" -gt 0 ]; then touch "$out/.done"; echo "ok $d: $n images"; else echo "FAILED $d (no images)"; fi
done
echo "DOWNLOADS FINISHED"
