#!/usr/bin/env python3
"""
Self-training data from real traffic video: for every vehicle the pipeline READ with high confidence
(confidence >= 0.95 and >= 3 valid readings), all of that vehicle's saved plate crops - including the
blurry / small ones the reader got wrong - get the confident plate number as their label.
    python plates/selftrain.py     (reads plates/video_self/*_plates.json + plates/dump/<video>/<tid>_<frame>.jpg)
Writes plates/ocr/train_self.csv = train_aug.csv + pseudo-labelled video crops (max 25 per vehicle).
Test plates / videos used for evaluation are never included (they are not in the self-training video list).
"""
import csv
import glob
import json
from pathlib import Path

P = Path(__file__).resolve().parent
rows, veh = [], 0
for jf in sorted(glob.glob(str(P / "video_self/*_plates.json"))):
    d = json.loads(Path(jf).read_text())
    stem = Path(d["video"]).stem
    for v in d["vehicles"]:
        if v["decision"] != "READ" or v["confidence"] < 0.95:
            continue
        crops = sorted(glob.glob(str(P / "dump" / stem / f"{v['vehicle']}_*.jpg")))
        if len(crops) < 3:
            continue
        veh += 1
        text = v["plate"].replace(" ", "")
        for c in crops[:25]:
            rows.append([str(Path("..") / Path(c).relative_to(P)), text, "Unknown"])
base = list(csv.reader(open(P / "ocr/train_aug.csv")))
csv.writer(open(P / "ocr/train_self.csv", "w", newline="")).writerows(base + rows)
print(f"{veh} confidently read vehicles -> {len(rows)} pseudo-labelled video crops; total {len(base) - 1 + len(rows)}")
