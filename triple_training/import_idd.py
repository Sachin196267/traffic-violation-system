#!/usr/bin/env python3
"""
Import still-image candidates from IDD (Indian Driving Dataset, dashcam view, on disk under
datasets/kaggle/idd) into the same unit format as the videos, so labelling / training / CV
treat them identically ("video" = idd_<scene group>, one crop per unit).

Selected motorcycles:
    all with >= 3 IDD rider boxes sitting on them       (triple candidates, mostly neighbours)
    a random sample with exactly 2 and exactly 1          (normal riding, hard negatives)
IDD file names encode the drive sequence; units from one sequence share a group so they never
end up in both train and test.
Writes triple_training/data/idd_<k>/{crops/*.jpg, units.json} for k = 0..4 (5 scene groups).
"""
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IDD = ROOT.parent / "datasets/kaggle/idd/IDDDetectionsYOLODataset"
N2, N1 = 500, 300
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


def on_bike(r, m):
    mw, mh = m[2] - m[0], m[3] - m[1]
    cx = (r[0] + r[2]) / 2
    return (m[0] - 0.15 * mw <= cx <= m[2] + 0.15 * mw and r[1] < m[1] + 0.3 * mh
            and m[1] - 0.2 * mh <= r[3] <= m[3] + 0.25 * mh)


by_n = defaultdict(list)
for lf in sorted(glob.glob(str(IDD / "*/labels/*.txt"))):
    rows = [l.split() for l in open(lf) if l.strip()]
    xyxy = lambda r: [float(r[1]) - float(r[3]) / 2, float(r[2]) - float(r[4]) / 2,
                      float(r[1]) + float(r[3]) / 2, float(r[2]) + float(r[4]) / 2]
    motos = [xyxy(r) for r in rows if r[0] == "6"]
    riders = [xyxy(r) for r in rows if r[0] == "8"]
    for m in motos:
        if m[3] - m[1] < 0.05:                            # too small to count riders
            continue
        att = [r for r in riders if on_bike(r, m)]
        by_n[min(len(att), 3)].append((lf, m, att))
print({k: len(v) for k, v in by_n.items()})
pick = by_n[3] + rng.sample(by_n[2], min(N2, len(by_n[2]))) + rng.sample(by_n[1], min(N1, len(by_n[1])))

groups = defaultdict(list)
for lf, m, att in pick:
    stem = Path(lf).stem
    seq = stem.rsplit("_", 1)[0] if "_" in stem else stem[:6]
    g = sum(map(ord, seq)) % 5
    groups[g].append((lf, m, att))

for g, items in groups.items():
    tag = f"idd_{g}"
    od = ROOT / "data" / tag
    (od / "crops").mkdir(parents=True, exist_ok=True)
    rows = []
    for lf, m, att in items:
        ip = lf.replace("/labels/", "/images/")[:-4]
        im = next((cv2.imread(ip + e) for e in (".jpg", ".png", ".jpeg") if Path(ip + e).exists()), None)
        if im is None:
            continue
        H, W = im.shape[:2]
        u = [min([m[0]] + [r[0] for r in att]), min([m[1]] + [r[1] for r in att]),
             max([m[2]] + [r[2] for r in att]), max([m[3]] + [r[3] for r in att])]
        uw, uh = u[2] - u[0], u[3] - u[1]
        x1, y1 = int(max(0, u[0] - 0.12 * uw) * W), int(max(0, u[1] - 0.08 * uh) * H)
        x2, y2 = int(min(1, u[2] + 0.12 * uw) * W), int(min(1, u[3] + 0.04 * uh) * H)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        uid = len(rows)
        name = f"{tag}_{uid:05d}.jpg"
        cv2.imwrite(str(od / "crops" / name), letterbox(im[y1:y2, x1:x2]), [cv2.IMWRITE_JPEG_QUALITY, 93])
        rows.append(dict(id=uid, video=tag, frame=0, t=float(uid), track=uid, box=[x1, y1, x2, y2],
                         moto=[round(m[0] * W, 1), round(m[1] * H, 1), round(m[2] * W, 1), round(m[3] * H, 1)],
                         crop_h=y2 - y1, n_rider_boxes=len(att),
                         n_heads=len(att), edge=False, frame_hw=[H, W], crop=name, source=Path(lf).stem))
    (od / "units.json").write_text(json.dumps(rows))
    print(tag, len(rows), "units")
