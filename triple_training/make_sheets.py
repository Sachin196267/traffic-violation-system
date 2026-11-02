#!/usr/bin/env python3
"""
Stage 2 - contact sheets for labelling motorcycle tracks by rider count, one video at a time.

Each cell = one motorcycle track, three crops spread over time (a rider hidden in one frame
is usually visible in another), headed by the cell number and track key "<video>:<track>".
Tracks are chosen by the cheap evidence from extract_units.py:
    A  some crop shows >= 3 heads or >= 3 rider boxes   -> all labelled
    B  best evidence == 2                                -> all labelled
    C  best evidence <= 1                                -> random --c-frac sample (catches hidden triples)

    python triple_training/make_sheets.py --video jaipur
Writes triple_training/sheets/<video>_<nn>.jpg + <video>_index.json.
Labels go to triple_training/labels/<video>.txt as "<video>:<track>=<label>" tokens:
    1, 2, 3 (three or more people on one two-wheeler), x (not a two-wheeler / broken crop),
    u (a two-wheeler, but the number of riders cannot be told).
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--per-sheet", type=int, default=20)
ap.add_argument("--c-frac", type=float, default=0.3, help="share of tier-C tracks to label")
ap.add_argument("--b-frac", type=float, default=1.0, help="share of tier-B tracks to label")
ap.add_argument("--cell", type=int, default=200)
ap.add_argument("--min-crops", type=int, default=2)
a = ap.parse_args()

tracks = defaultdict(list)
for f in [ROOT / "data" / a.video / "units.json"]:
    for r in json.loads(f.read_text()):
        tracks[f"{r['video']}:{r['track']}"].append(r)


def evidence(rows):
    return max(max(r["n_heads"], r["n_rider_boxes"]) for r in rows)


def tier(rows):
    e = evidence(rows)
    return "A" if e >= 3 else "B" if e == 2 else "C"


done = set()
for lf in (ROOT / "labels").glob("*.txt"):
    for tok in lf.read_text().split():
        done.add(tok.split("=")[0])
by_tier = {"A": [], "B": [], "C": []}
for k, rows in tracks.items():
    if len(rows) >= a.min_crops and k not in done:
        by_tier[tier(rows)].append(k)
rnd = random.Random(0)
rnd.shuffle(by_tier["C"])
rnd.shuffle(by_tier["B"])
nC = min(len(by_tier["C"]), max(10, round(a.c_frac * len(by_tier["C"]))))
nB = min(len(by_tier["B"]), max(10, round(a.b_frac * len(by_tier["B"]))))
keys = by_tier["A"] + by_tier["B"][:nB] + by_tier["C"][:nC]
keys.sort(key=lambda k: int(k.split(":")[1]))
print(f"{a.video}: A={len(by_tier['A'])} B={nB}/{len(by_tier['B'])} C={nC}/{len(by_tier['C'])} -> {len(keys)} tracks")


def pick3(rows):
    rows = sorted(rows, key=lambda r: r["t"])
    if len(rows) <= 3:
        return rows
    big = sorted(rows, key=lambda r: -r["crop_h"])[:max(3, len(rows) // 2)]   # prefer the larger half
    big.sort(key=lambda r: r["t"])
    idx = np.linspace(0, len(big) - 1, 3).round().astype(int)
    return [big[i] for i in idx]


out = ROOT / "sheets"
out.mkdir(exist_ok=True)
C = a.cell
index = {}
single = all(len(tracks[k]) == 1 for k in keys)
if single:                                               # still images: dense 6-column grid
    per = 36
    for s in range(0, len(keys), per):
        chunk = keys[s:s + per]
        rows_n = (len(chunk) + 5) // 6
        sheet = np.full((rows_n * (C + 22), 6 * (C + 6), 3), 255, np.uint8)
        for i, k in enumerate(chunk):
            r0, c0 = (i // 6) * (C + 22), (i % 6) * (C + 6)
            im = cv2.imread(str(ROOT / "data" / tracks[k][0]["video"] / "crops" / tracks[k][0]["crop"]))
            sheet[r0 + 20:r0 + 20 + C, c0:c0 + C] = cv2.resize(im, (C, C))
            cv2.putText(sheet, f"{i + 1:02d}", (c0 + 2, r0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        name = f"{a.video}_{s // per:02d}"
        cv2.imwrite(str(out / f"{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        index[name] = chunk
    (out / f"{a.video}_index.json").write_text(json.dumps(index, indent=0))
    print(f"{len(index)} sheets written (single-image grid)")
    raise SystemExit
for s in range(0, len(keys), a.per_sheet):
    chunk = keys[s:s + a.per_sheet]
    cols = 2
    rows_n = (len(chunk) + cols - 1) // cols
    sheet = np.full((rows_n * (C + 26), cols * (3 * C + 12), 3), 255, np.uint8)
    for i, k in enumerate(chunk):
        r0, c0 = (i // cols) * (C + 26), (i % cols) * (3 * C + 12)
        cv2.putText(sheet, f"{i + 1:02d}  {k}", (c0 + 4, r0 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        for j, r in enumerate(pick3(tracks[k])):
            im = cv2.imread(str(ROOT / "data" / r["video"] / "crops" / r["crop"]))
            sheet[r0 + 24:r0 + 24 + C, c0 + j * C:c0 + (j + 1) * C] = cv2.resize(im, (C, C))
    name = f"{a.video}_{s // a.per_sheet:02d}"
    cv2.imwrite(str(out / f"{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    index[name] = chunk
(out / f"{a.video}_index.json").write_text(json.dumps(index, indent=0))
print(f"{len(index)} sheets written")
