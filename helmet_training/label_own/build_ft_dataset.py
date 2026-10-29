#!/usr/bin/env python3
"""
Fine-tune dataset = original clean head dataset (symlinked) + own-video rider crops (labelled in labels.txt).

Own crops: labelled heads -> boxes (0 helmet, 1 no_helmet); rejected candidates ('x') are greyed out (fill 114)
so the model is not taught that they are background. Own train crops are repeated OVERSAMPLE times.
All own train crops go to train (the bareheaded riders sit in few 10 s blocks, so an own-val split starved train);
validation = original clean val set; own test split is untouched and the final epoch (last.pt) is used.
Output: datasets/helmet_nohelmet_ft_own/  (+ own_test/ for evaluation)
"""
import json, os, shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datasets/helmet_nohelmet_final"
LO = ROOT / "helmet_training/label_own"
DST = ROOT / "datasets/helmet_nohelmet_ft_own"
OVERSAMPLE = 3

shutil.rmtree(DST, ignore_errors=True)
for sp in ("train", "val", "test"):
    (DST / "images" / sp).mkdir(parents=True)
    (DST / "labels" / sp).mkdir(parents=True)
    for im in (SRC / "images" / sp).iterdir():
        os.symlink(im, DST / "images" / sp / im.name)
        lb = SRC / "labels" / sp / (im.stem + ".txt")
        if lb.exists():
            shutil.copy(lb, DST / "labels" / sp / lb.name)
(DST / "own_test/images").mkdir(parents=True)
(DST / "own_test/labels").mkdir(parents=True)

cands = json.loads((LO / "candidates_labelled.json").read_text())
by_crop = defaultdict(list)
for c in cands:
    by_crop[c["crop"]].append(c)

stats = Counter()
for crop, cs in by_crop.items():
    split = cs[0]["split"]
    if split == "drop" or all(c["label"] == "x" for c in cs):
        continue
    video, fi = cs[0]["video"], int(crop.split("_")[1])
    im = cv2.imread(str(LO / "crops" / crop))
    H, W = im.shape[:2]
    lines = []
    for c in cs:
        x1, y1, x2, y2 = c["box"]
        if c["label"] == "x":
            im[int(y1):int(y2), int(x1):int(x2)] = 114
            continue
        k = 0 if c["label"] == "h" else 1
        lines.append(f"{k} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        stats[(split, c["label"])] += 1
    if split == "test":
        dirs, reps = DST / "own_test", 1
    else:
        dirs, reps = DST, OVERSAMPLE if split == "train" else 1
    for r in range(reps):
        stem = f"own_{Path(crop).stem}_r{r}"
        sub = "" if split == "test" else split
        cv2.imwrite(str(dirs / "images" / sub / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (dirs / "labels" / sub / f"{stem}.txt").write_text("\n".join(lines))
    stats[(split, "crops")] += 1

(DST / "data.yaml").write_text(
    f"# old clean head dataset + own-video rider crops (labels: helmet_training/label_own/labels.txt)\n"
    f"path: {DST}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames:\n  0: helmet\n  1: no_helmet\n")
(DST / "own_test.yaml").write_text(f"path: {DST}\ntrain: own_test/images\nval: own_test/images\nnc: 2\nnames:\n  0: helmet\n  1: no_helmet\n")
report = {f"{a}/{b}": v for (a, b), v in sorted(stats.items())}
(DST / "build_report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
