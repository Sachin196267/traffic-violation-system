#!/usr/bin/env python3
"""
Import the staged Roboflow set (datasets/external/abuzarkhaaan_helmetbehncode, CC BY 4.0:
one red motorcycle, the same few people, 1 / 2 / 3 on the bike) as TRAINING-ONLY units.
Its labels come from the dataset's own Bike/Passenger boxes (3-passenger cases checked by eye).
The video name starts with "train_only_" so build_dataset.py never puts it in a test fold -
a single staged scene would make the held-out numbers look better than they are.
Writes triple_training/data/train_only_rf/{crops/*.jpg, units.json} and labels/train_only_rf.txt.
"""
import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "datasets/external/abuzarkhaaan_helmetbehncode"
TAG = "train_only_rf"
N_PER = {1: 150, 2: 150, 3: 1000}
rng = random.Random(0)


def letterbox(img, size=256):
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                   interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    out = np.full((size, size, 3), 114, np.uint8)
    y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


cands = {1: [], 2: [], 3: []}
for lf in sorted(glob.glob(str(SRC / "*/labels/*.txt"))):
    rows = [l.split() for l in open(lf) if l.strip()]
    bikes = [list(map(float, r[1:5])) for r in rows if r[0] == "0"]
    pas = [list(map(float, r[1:5])) for r in rows if r[0] == "2"]
    for b in bikes:
        bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
        P = [p for p in pas if bx1 - 0.1 * b[2] <= p[0] <= bx2 + 0.1 * b[2] and by1 - 0.6 * b[3] <= p[1] <= by2]
        if 1 <= len(P) <= 3:
            cands[len(P)].append((lf, (bx1, by1, bx2, by2), P))

od = ROOT / "data" / TAG
(od / "crops").mkdir(parents=True, exist_ok=True)
units, labels = [], []
for n, items in cands.items():
    rng.shuffle(items)
    for lf, (bx1, by1, bx2, by2), P in items[:N_PER[n]]:
        ip = lf.replace("/labels/", "/images/")[:-4]
        im = next((cv2.imread(ip + e) for e in (".jpg", ".jpeg", ".png") if Path(ip + e).exists()), None)
        if im is None:
            continue
        H, W = im.shape[:2]
        xs = [bx1, bx2] + [p[0] - p[2] / 2 for p in P] + [p[0] + p[2] / 2 for p in P]
        ys = [by1, by2] + [p[1] - p[3] / 2 for p in P] + [p[1] + p[3] / 2 for p in P]
        uw, uh = max(xs) - min(xs), max(ys) - min(ys)
        x1, y1 = int(max(0, min(xs) - 0.12 * uw) * W), int(max(0, min(ys) - 0.08 * uh) * H)
        x2, y2 = int(min(1, max(xs) + 0.12 * uw) * W), int(min(1, max(ys) + 0.04 * uh) * H)
        uid = len(units)
        name = f"{TAG}_{uid:05d}.jpg"
        cv2.imwrite(str(od / "crops" / name), letterbox(im[y1:y2, x1:x2]), [cv2.IMWRITE_JPEG_QUALITY, 93])
        units.append(dict(id=uid, video=TAG, frame=0, t=float(uid), track=uid, box=[x1, y1, x2, y2],
                          moto=[round(bx1 * W, 1), round(by1 * H, 1), round(bx2 * W, 1), round(by2 * H, 1)],
                          crop_h=y2 - y1, n_rider_boxes=len(P), n_heads=len(P), edge=False,
                          frame_hw=[H, W], crop=name, source=Path(lf).stem))
        labels.append(f"{TAG}:{uid}={n}")
(od / "units.json").write_text(json.dumps(units))
(ROOT / "labels" / f"{TAG}.txt").write_text(" ".join(labels) + "\n")
print({n: sum(l.endswith(f"={n}") for l in labels) for n in (1, 2, 3)})
