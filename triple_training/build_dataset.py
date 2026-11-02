#!/usr/bin/env python3
"""
Stage 3 - classifier dataset from the track labels.

Every crop of a labelled track inherits the track label (labels/<video>.txt), unless
labels/<video>.crops.txt overrides that crop (e.g. a third rider hidden in some frames):
    1 -> one, 2 -> two, 3 -> three_plus, x -> not_bike ;  u (uncountable) is left out.
Folds are split by VIDEO (never the same video in train and held-out), balanced so every
fold holds some triple-riding tracks, so the cross-validation measures unseen cameras.
Sources named "train_only_*" (staged / single-scene sets) are always in training, never scored.
three_plus is rare, so its training crops are repeated (links) and degraded copies
(low bitrate, blur, noise, dark) are added for every class to cover poor cameras.

Writes triple_training/samples.json and triple_training/cls/{fold0..4,all}/{train,val}/<class>/.
"""
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MAP = {"1": "one", "2": "two", "3": "three_plus", "x": "not_bike"}
K = 5
REPEAT_TRIPLE = 4
rng = random.Random(0)

labels, crop_labels = {}, {}
for lf in sorted((ROOT / "labels").glob("*.txt")):
    for tok in lf.read_text().split():
        k, v = tok.split("=")
        (crop_labels if lf.name.endswith(".crops.txt") else labels)[k] = v
units = defaultdict(list)
for f in sorted((ROOT / "data").glob("*/units.json")):
    for r in json.loads(f.read_text()):
        units[f"{r['video']}:{r['track']}"].append(r)

# fold assignment: greedy by triple count so each fold gets a share of positives
TPT_SITES = {  # Tirupati clips grouped by the junction they were filmed at (checked by eye)
    "tpt-a": {"tpt-01", "tpt-02", "tpt-03", "tpt-04", "tpt-05", "tpt-06", "tpt-07", "tpt-08", "tpt-09",
              "tpt-10", "tpt-11", "tpt-12", "tpt-13", "tpt-14", "tpt-15", "tpt-16", "tpt-17", "tpt-18",
              "tpt-20", "tpt-21"},
    "tpt-b": {"tpt-22", "tpt-23"},
    "tpt-c": {"tpt-25", "tpt-26", "tpt-27"},
}


def group(video):
    """Clips from one camera site share a fold: 'hydpol-04' -> 'hydpol', Tirupati by site."""
    for g, vids in TPT_SITES.items():
        if video in vids:
            return g
    return video.split("-")[0] if "-" in video else video


by_video = defaultdict(Counter)
for k, v in labels.items():
    by_video[group(k.split(":")[0])][v] += 1
for c, v in crop_labels.items():
    by_video[group(c.rsplit("_", 1)[0])][v] += 1
videos = sorted(by_video, key=lambda v: (-by_video[v]["3"], -sum(by_video[v].values())))
fold_of, load = {}, [[0, 0] for _ in range(K)]
for v in [v for v in videos if v.startswith("train_only_")]:
    fold_of[v] = -1                                     # never held out, never scored
scored = [v for v in videos if not v.startswith("train_only_")]
for v in [v for v in scored if by_video[v]["3"]]:          # spread the positives first
    f = min(range(K), key=lambda i: (load[i][0], load[i][1]))
    fold_of[v] = f
    load[f][0] += by_video[v]["3"]
    load[f][1] += sum(by_video[v].values())
for v in [v for v in scored if not by_video[v]["3"]]:      # then balance the rest by size
    f = min(range(K), key=lambda i: load[i][1])
    fold_of[v] = f
    load[f][1] += sum(by_video[v].values())
print("fold loads [triples, tracks]:", load)

samples = []
for k in sorted(set(labels) | {f"{r['video']}:{r['track']}" for rows in units.values() for r in rows
                               if r["crop"] in crop_labels}):
    video = k.split(":")[0]
    v = labels.get(k, "u")
    for r in units[k]:
        lab = crop_labels.get(r["crop"], v)             # per-crop override (ID switch, occluded rider)
        if lab not in MAP:
            continue
        # crops relabelled individually form their own scoring unit, so a tracker ID switch
        # never mixes two different motorcycles inside one track decision
        key = k if r["crop"] not in crop_labels else f"{k}:{lab}"
        samples.append(dict(key=key, video=video, label=MAP[lab], crop_h=r["crop_h"], t=r["t"],
                            n_heads=r["n_heads"], n_rider_boxes=r["n_rider_boxes"], fold=fold_of[group(video)],
                            path=str(ROOT / "data" / video / ("crops_clean" if (ROOT / "data" / video / "crops_clean").exists()
                                                              else "crops") / r["crop"])))
(ROOT / "samples.json").write_text(json.dumps(samples))
print(len(samples), "crops;", Counter(s["label"] for s in samples))


def degrade(im):
    h, w = im.shape[:2]
    s = rng.uniform(0.25, 0.6)
    im = cv2.resize(cv2.resize(im, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA),
                    (w, h), interpolation=cv2.INTER_LINEAR)
    r = rng.random()
    if r < 0.3:
        k = rng.choice([3, 5, 7])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1)
        im = cv2.filter2D(im, -1, cv2.warpAffine(kern, M, (k, k)))
    elif r < 0.5:
        im = cv2.GaussianBlur(im, (3, 3), 0)
    if rng.random() < 0.3:                                 # dusk / night look
        im = np.clip(im.astype(np.float32) * rng.uniform(0.35, 0.7), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        im = np.clip(im + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(4, 12), im.shape),
                     0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(15, 45)])
    return cv2.imdecode(buf, 1)


aug_dir = ROOT / "aug"
aug_dir.mkdir(exist_ok=True)
deg = {}
for s in samples:
    d = aug_dir / f"deg_{Path(s['path']).name}"
    if not d.exists():
        cv2.imwrite(str(d), degrade(cv2.imread(s["path"])))
    deg[s["path"]] = d

cls = ROOT / "cls"
if cls.exists():
    shutil.rmtree(cls)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src)


for name, held in [(f"fold{i}", i) for i in range(K)] + [("all", None)]:
    for s in samples:
        split = "val" if s["fold"] == held else "train"
        if name == "all":
            split = "val" if s["fold"] == 0 and rng.random() < 0.1 else "train"   # tiny val, only for monitoring
        base = cls / name / split / s["label"]
        p = Path(s["path"])
        link(p, base / p.name)
        if split == "train":
            link(deg[s["path"]], base / deg[s["path"]].name)
            if s["label"] == "three_plus" and not s["video"].startswith("train_only_"):   # real ones only
                for j in range(1, REPEAT_TRIPLE):
                    link(p, base / f"rep{j}_{p.name}")
                    link(deg[s["path"]], base / f"rep{j}_{deg[s['path']].name}")
print("folders written under", cls)
