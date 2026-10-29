#!/usr/bin/env python3
"""
Import TRAINING-ONLY triple-riding positives from the Kaggle dataset
devgurucodes/trafffic-violations-triple-riding-no-helmet-plate (downloaded with the user's OK to
datasets/external/devgurucodes_triple_riding; no license stated -> project/study use only).

Only the original images ("dst_*") are used - the "aug_*" files are augmented copies of them.
Each TripleRiding box (class 3) >= --min-h px becomes one unit, cropped like the video units
(box + 12 % side / 8 % top padding, letterboxed to 256).  Source name "train_only_dg" keeps it
out of every test fold.  Writes data/train_only_dg/ and labels/train_only_dg.txt.
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "datasets/external/devgurucodes_triple_riding/master_traffic_violation_dataset"
TAG = "train_only_dg"
ap = argparse.ArgumentParser()
ap.add_argument("--min-h", type=int, default=80)
a = ap.parse_args()


def letterbox(img, size=256):
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                   interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    out = np.full((size, size, 3), 114, np.uint8)
    y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


od = ROOT / "data" / TAG
(od / "crops").mkdir(parents=True, exist_ok=True)
units, labels, skipped = [], [], 0
for lf in sorted(glob.glob(str(SRC / "*/labels/dst_*.txt"))):
    ip = next((p for p in glob.glob(lf.replace("/labels/", "/images/")[:-4] + ".*")), None)
    im = cv2.imread(ip) if ip else None
    if im is None:
        continue
    H, W = im.shape[:2]
    for l in open(lf):
        r = l.split()
        if not r or r[0] != "3":
            continue
        x, y, w, h = map(float, r[1:5])
        if h * H < a.min_h:
            skipped += 1
            continue
        x1, y1 = int(max(0, x - w / 2 - 0.12 * w) * W), int(max(0, y - h / 2 - 0.08 * h) * H)
        x2, y2 = int(min(1, x + w / 2 + 0.12 * w) * W), int(min(1, y + h / 2 + 0.04 * h) * H)
        uid = len(units)
        name = f"{TAG}_{uid:05d}.jpg"
        cv2.imwrite(str(od / "crops" / name), letterbox(im[y1:y2, x1:x2]), [cv2.IMWRITE_JPEG_QUALITY, 93])
        units.append(dict(id=uid, video=TAG, frame=0, t=float(uid), track=uid, box=[x1, y1, x2, y2],
                          crop_h=y2 - y1, n_rider_boxes=3, n_heads=3, edge=False, frame_hw=[H, W],
                          crop=name, source=Path(lf).stem))
        labels.append(f"{TAG}:{uid}=3")
(od / "units.json").write_text(json.dumps(units))
(ROOT / "labels" / f"{TAG}.txt").write_text(" ".join(labels) + "\n")
print(len(units), "triple units;", skipped, "boxes too small skipped")
