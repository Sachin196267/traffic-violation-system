#!/usr/bin/env python3
"""
Record labels for one contact sheet, in cell order:
    python triple_training/label.py blr_pov_00 "x x x 2 x x x x x x x x 1 x 1 x x x 1 u"
Labels: 1, 2, 3 (three or more people on the two-wheeler), x (every crop shows something that
is not a ridden two-wheeler: parked bike, own handlebars, rickshaw, car), u (can't count the
riders, OR a mixed track whose crops show different things - u is left out of training, x is not).
Appends "<video>:<track>=<label>" tokens to triple_training/labels/<video>.txt (re-labelling a
track overwrites its previous label).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sheet, labs = sys.argv[1], sys.argv[2].split()
video = sheet.rsplit("_", 1)[0]
keys = json.loads((ROOT / "sheets" / f"{video}_index.json").read_text())[sheet]
crops = video.endswith("C") and keys and keys[0].endswith(".jpg")     # crop-level sheet (crop_sheets.py)
if crops:
    video = video[:-1]
if len(keys) != len(labs):
    sys.exit(f"{sheet} has {len(keys)} cells, got {len(labs)} labels")
bad = [l for l in labs if l not in {"1", "2", "3", "x", "u"}]
if bad:
    sys.exit(f"unknown labels {bad}")
lf = ROOT / "labels" / (f"{video}.crops.txt" if crops else f"{video}.txt")
cur = dict(t.split("=") for t in lf.read_text().split()) if lf.exists() else {}
cur.update(zip(keys, labs))
lf.write_text(" ".join(f"{k}={v}" for k, v in cur.items()) + "\n")
print(f"{sheet}: {len(labs)} labels -> {lf.name} ({len(cur)} total); "
      f"triples here: {[k for k, l in zip(keys, labs) if l == '3']}")
