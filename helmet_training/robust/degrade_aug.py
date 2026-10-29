#!/usr/bin/env python3
"""
Add degraded copies of every training patch to cls/<fold>/train (val/held-out videos untouched):
low-bitrate look (downscale x0.25-0.6 + JPEG q15-45), motion/gaussian blur, noise, and for helmet
patches a hue rotation (so yellow / red / white helmets are as familiar as black ones).
Fixes the unseen-video failure where coloured helmets in grainy footage were called no_helmet.
Run after build_dataset.py.
"""
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
AUG = ROOT / "aug_patches"
AUG.mkdir(exist_ok=True)
rng = random.Random(0)


def degrade(im, cls):
    h, w = im.shape[:2]
    if cls == "helmet" and rng.random() < 0.6:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + rng.randint(20, 160)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.8, 1.6), 0, 255)
        im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    s = rng.uniform(0.25, 0.6)
    im = cv2.resize(cv2.resize(im, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    r = rng.random()
    if r < 0.3:
        k = rng.choice([3, 5, 7])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1)
        im = cv2.filter2D(im, -1, cv2.warpAffine(kern, M, (k, k)))
    elif r < 0.5:
        im = cv2.GaussianBlur(im, (3, 3), 0)
    if rng.random() < 0.4:
        im = np.clip(im + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(4, 12), im.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(15, 45)])
    return cv2.imdecode(buf, 1)


made = {}
for fold_dir in sorted((ROOT / "cls").iterdir()):
    n = 0
    for cls_dir in (fold_dir / "train").iterdir():
        for f in list(cls_dir.glob("*.jpg")):
            if f.name.startswith("deg_"):
                continue
            src = f.resolve()
            dst = AUG / cls_dir.name / f"deg_{src.stem}.jpg"
            if dst not in made:
                dst.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(dst), degrade(cv2.imread(str(src)), cls_dir.name))
                made[dst] = 1
            link = cls_dir / dst.name
            if not link.exists():
                link.symlink_to(dst)
                n += 1
    print(fold_dir.name, "added", n)
