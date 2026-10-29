#!/usr/bin/env python3
"""
Fine-tune the light-state model (LISA/TVD, Kaggle) with Indian signal-head crops, testing on held-out clips.

Indian data: crops labelled by eye (head_labels.json; 'unsure' dropped) + for every head labelled not_light in all
its sampled frames (signboards, red cars, overlay circles...) 30 more crops of that head from heads.npz (a sign
never turns into a signal). Clips are split into two groups; model A is fine-tuned without group A's clips and
tested on them, model B likewise; the final model uses everything.
Output: runs/signal_state_ft/{A,B,final}/weights/last.pt, runs/signal_state_ft/report.json.
"""
import json, os, random, shutil, zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "signal_training"
OUT = ROOT / "runs/signal_state_ft"; OUT.mkdir(parents=True, exist_ok=True)
BASE = S / "base_crops"
C = ["red", "yellow", "green", "not_light"]
rng = random.Random(0)
if not (BASE / "train").exists():
    zipfile.ZipFile(ROOT / "runs/signal_state_dl/out/tl_crops.zip").extractall(BASE)

L = json.loads((S / "head_labels.json").read_text())
idx = dict(map(tuple, json.loads((S / "india_index.json").read_text())))
IND = S / "india_all"; IND.mkdir(exist_ok=True)
items = []   # (file, label, clip)
per_head = defaultdict(list)
for k, lab in L.items():
    if lab == "unsure":
        continue
    clip, hid, i = k.rsplit(":", 2)[0].split(":")[0], ":".join(k.split(":")[1:3]), k.split(":")[-1]
    items.append((str(S / "india_crops" / idx[k]), lab, clip))
    per_head[(clip, hid)].append(lab)
for (clip, hid), labs in per_head.items():
    if labs and all(l == "not_light" for l in labs) and len(labs) >= 3:
        npz = np.load(ROOT / "red_light/runs" / clip / "heads.npz")
        crops = npz[hid.replace(":", "_")]
        for j in rng.sample(range(len(crops)), min(30, len(crops))):
            fn = IND / f"{clip}_{hid.replace(':', '_')}_{j}.jpg"
            cv2.imwrite(str(fn), crops[j])
            items.append((str(fn), "not_light", clip))
clips = sorted({c for _, _, c in items})
rng.shuffle(clips)
groups = {"A": set(clips[::2]), "B": set(clips[1::2])}
print("Indian items", Counter((l) for _, l, _ in items), "clips", len(clips), flush=True)


def build(name, held):
    d = OUT / f"data_{name}"
    if d.exists():
        shutil.rmtree(d)
    for sp in ("train", "val"):
        for c in C:
            (d / sp / c).mkdir(parents=True)
    for sp in ("train", "val"):
        for c in C:
            for f in (BASE / sp / c).iterdir():
                os.symlink(f, d / sp / c / f.name)
    for k, (f, lab, clip) in enumerate(items):
        if clip in held:
            continue
        for r in range(5):                       # few Indian crops: repeat them
            os.symlink(f, d / "train" / lab / f"in{r}_{k}.jpg")
    return d


report = {}
for name in ("A", "B", "final"):
    held = groups.get(name, set())
    wp = OUT / name / "weights/last.pt"
    if not wp.exists():
        d = build(name, held)
        m = YOLO(str(ROOT / "runs/signal_state/weights/last.pt"))
        m.train(data=str(d), imgsz=64, epochs=6, batch=128, device="mps", workers=2, project=str(OUT), name=name,
                exist_ok=True, optimizer="AdamW", lr0=3e-4, cos_lr=True, fliplr=0.5, hsv_h=0.0, hsv_s=0.3, hsv_v=0.4,
                degrees=3, scale=0.2, translate=0.1, erasing=0.1, auto_augment=None, seed=0, plots=False, verbose=False)
    if not held:
        continue
    m = YOLO(str(wp))
    test = [(f, l) for f, l, c in items if c in held]
    conf = defaultdict(Counter)
    for (f, l), r in zip(test, m.predict([f for f, _ in test], imgsz=64, verbose=False)):
        conf[l][r.names[int(r.probs.top1)] + ("" if float(r.probs.top1conf) >= 0.9 else "?")] += 1
    report[name] = {"held_clips": sorted(held), "confusion": {k: dict(v) for k, v in conf.items()}}
    print(name, json.dumps(report[name]["confusion"]), flush=True)
(OUT / "report.json").write_text(json.dumps(report, indent=1))
print("FINETUNE DONE", flush=True)
