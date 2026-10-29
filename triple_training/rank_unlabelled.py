#!/usr/bin/env python3
"""
Active learning: score every not-yet-labelled crop with the current model and build crop-level
labelling sheets from
    * the highest p(three_plus) crops  -> likely new triples, and the model's worst false alarms
    * a random sample of the rest      -> keeps the negatives unbiased
    python triple_training/rank_unlabelled.py --videos indore pune bombay vns_cross --top 72 --random 36
Sheets are named <video>C_<nn> and recorded with label.py like other crop sheets.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

R = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--videos", nargs="+", required=True)
ap.add_argument("--weights", default=str(R.parent / "runs/triple_rider/final/weights/last.pt"))
ap.add_argument("--top", type=int, default=72)
ap.add_argument("--random", type=int, default=36)
ap.add_argument("--min-h", type=int, default=80)
a = ap.parse_args()

m = YOLO(a.weights)
done = set()
for lf in (R / "labels").glob("*.txt"):
    for t in lf.read_text().split():
        done.add(t.split("=")[0])
rng = random.Random(0)
for v in a.videos:
    U = json.loads((R / "data" / v / "units.json").read_text())
    todo = [r for r in U if r["crop"] not in done and f"{v}:{r['track']}" not in done and r["crop_h"] >= a.min_h]
    paths = [str(R / "data" / v / "crops" / r["crop"]) for r in todo]
    k3 = [k for k, n in m.names.items() if n == "three_plus"][0]
    p3 = []
    for i in range(0, len(paths), 256):
        for res in m.predict(paths[i:i + 256], imgsz=224, verbose=False):
            p3.append(float(res.probs.data[k3]))
    order = np.argsort(p3)[::-1]
    top = [todo[i] for i in order[:a.top]]
    rest = [todo[i] for i in order[a.top:]]
    rng.shuffle(rest)
    pick = top + rest[:a.random]
    index, C = {}, 200
    old = R / "sheets" / f"{v}C_index.json"
    start = len(json.loads(old.read_text())) if old.exists() else 0
    if old.exists():
        index = json.loads(old.read_text())
    for s in range(0, len(pick), 36):
        chunk = pick[s:s + 36]
        rows = (len(chunk) + 5) // 6
        sheet = np.full((rows * (C + 22), 6 * (C + 6), 3), 255, np.uint8)
        for i, r in enumerate(chunk):
            r0, c0 = (i // 6) * (C + 22), (i % 6) * (C + 6)
            sheet[r0 + 20:r0 + 20 + C, c0:c0 + C] = cv2.resize(cv2.imread(str(R / "data" / v / "crops" / r["crop"])), (C, C))
            cv2.putText(sheet, f"{i + 1:02d}", (c0 + 2, r0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        name = f"{v}C_{start + s // 36:02d}"
        cv2.imwrite(str(R / "sheets" / f"{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        index[name] = [r["crop"] for r in chunk]
    old.write_text(json.dumps(index))
    print(f"{v}: {len(todo)} unlabelled crops; top p3 = {[round(p3[i], 2) for i in order[:5]]}; "
          f"sheets {start:02d}..{start + (len(pick) - 1) // 36:02d}")
