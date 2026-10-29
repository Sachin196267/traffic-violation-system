#!/usr/bin/env python3
"""
Build classifier datasets (helmet / no_helmet / not_head) from labelled persons + public head boxes.

  * own videos: frames of a labelled person within 3 s of the labelled view (and >= half its size) inherit the label (u = can't tell -> left out)
  * public: head boxes of datasets/helmet_nohelmet_final (train split), cut with the same geometry as the video patches (head x2 square). Public data is train-only.
    (abuzarkhaaan was tried and dropped: its "Helmet" boxes cover whole riders/images.)

Leave-videos-out folds: videos are split into K groups; fold k trains on the rest and tests on group k.
Output (symlinks): cls/fold{k}/{train,val}/{helmet,no_helmet,not_head}/*.jpg and cls/all/... for the
final model; samples.json lists every own sample with its person, label, fold.
"""
import argparse, json, random, xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parents[1]
DATA, LABELS, CLS = ROOT / "data", ROOT / "labels", ROOT / "cls"
PUB = ROOT / "public_patches"
NAMES = {"h": "helmet", "n": "no_helmet", "x": "not_head"}
P = 128
# clips cut from the same source footage must share a fold
SAME_SOURCE = {"mum_traffic": "mumbai"}
PROPAGATE_S = 3.0  # copy a person's label only to frames this close to the labelled view

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def patch(img, box):
    H, W = img.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(box[2] - box[0], box[3] - box[1])
    x1, y1, x2, y2 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    pad = [max(0, -y1), max(0, y2 - H), max(0, -x1), max(0, x2 - W)]
    p = cv2.copyMakeBorder(img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)], *pad, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(p, (P, P), interpolation=cv2.INTER_CUBIC)


# ---- public patches (built once) ----
if not PUB.exists():
    for c in NAMES.values():
        (PUB / c).mkdir(parents=True, exist_ok=True)
    src = PROJ / "datasets/helmet_nohelmet_final"
    n = 0
    for lab in (src / "labels/train").glob("*.txt"):
        imgs = list((src / "images/train").glob(lab.stem + ".*"))
        if not imgs or not lab.read_text().strip():
            continue
        im = cv2.imread(str(imgs[0]))
        H, W = im.shape[:2]
        for i, line in enumerate(lab.read_text().split("\n")):
            if not line.strip():
                continue
            c, x, y, w, h = map(float, line.split())
            if min(w * W, h * H) < 10:
                continue
            b = [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
            cv2.imwrite(str(PUB / ("helmet" if c == 0 else "no_helmet") / f"hnf_{lab.stem}_{i}.jpg"), patch(im, b))
            n += 1
    print("public patches", n, {c: len(list((PUB / c).glob("*.jpg"))) for c in NAMES.values()})

# ---- own labelled samples ----
samples = []
for vd in sorted(DATA.iterdir()):
    lf = LABELS / f"{vd.name}.txt"
    if not lf.exists() or not (vd / "persons.json").exists():
        continue
    lab = {}
    for tok in lf.read_text().split():
        if ":" in tok:
            k, v = tok.split(":")
            lab[int(k)] = v
    rows = {r["id"]: r for r in json.loads((vd / "heads.json").read_text())}
    for p in json.loads((vd / "persons.json").read_text()):
        l = lab.get(p["pid"])
        if l is None:
            continue
        rep = rows[p["rep"]]
        for m in p["members"]:
            r = rows[m]
            # tracks can switch identity: only trust frames near the labelled view, at a similar scale
            if abs(r["t"] - rep["t"]) > PROPAGATE_S or r["head_px"] < 0.5 * rep["head_px"]:
                continue
            samples.append({"video": vd.name, "pid": p["pid"], "track": p["track"], "label": l, "head_px": r["head_px"],
                            "rep": m == p["rep"], "path": str(vd / "patches" / r["patch"])})

for s in samples:
    s["group"] = SAME_SOURCE.get(s["video"], s["video"])
videos = sorted({s["group"] for s in samples})
# balance folds by no_helmet persons, then by total persons
nh = Counter(s["group"] for s in samples if s["label"] == "n" and s["rep"])
tot = Counter(s["group"] for s in samples if s["rep"])
random.Random(args.seed).shuffle(videos)
fold_of, load = {}, [[0, 0] for _ in range(args.folds)]
for v in sorted(videos, key=lambda v: (-nh[v], -tot[v])):
    k = min(range(args.folds), key=lambda k: load[k])
    fold_of[v] = k
    load[k][0] += nh[v]
    load[k][1] += tot[v]
for s in samples:
    s["fold"] = fold_of[s["group"]]
(ROOT / "samples.json").write_text(json.dumps(samples))


def link(dst, src):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src)


import shutil
if CLS.exists():
    shutil.rmtree(CLS)
pub = [(c, f) for c in NAMES.values() for f in (PUB / c).glob("*.jpg")]
for k in list(range(args.folds)) + ["all"]:
    d = CLS / (f"fold{k}" if k != "all" else "all")
    for c, f in pub:
        link(d / "train" / c / f.name, f)
    for s in samples:
        if s["label"] not in NAMES:
            continue
        split = "val" if k != "all" and s["fold"] == k else "train"
        if k == "all" and s["fold"] == 0:
            link(d / "val" / NAMES[s["label"]] / Path(s["path"]).name, Path(s["path"]))  # nominal val (not used for selection)
        link(d / split / NAMES[s["label"]] / Path(s["path"]).name, Path(s["path"]))

print("videos", len(videos), "folds (no_helmet persons, persons):", load)
print("own samples", Counter(s["label"] for s in samples), "persons", Counter(s["label"] for s in samples if s["rep"]))
