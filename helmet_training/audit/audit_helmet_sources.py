#!/usr/bin/env python3
"""Read-only integrity + duplicate audit of every helmet dataset in the project.

Writes helmet_training/audit/source_audit.json and prints a summary.
Never modifies source datasets.
"""
import json, os, sys, collections
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "ph": (ROOT / "Motorcycle Helmet Detection Dataset PH.v2i.yolov8", ["train", "valid", "test"], "split/images"),
    "bike": (ROOT / "Bike Helmet Detection.v2-more-preprocessing-augmentation.yolov8", ["train", "valid", "test"], "split/images"),
    "india": (ROOT / "datasets/kaggle/helmet_india", ["train", "valid"], "split/images"),
    "merged10k": (ROOT / "datasets/merged/helmet_traffic_10k", ["train", "val"], "images/split"),
}
EXT = {".jpg", ".jpeg", ".png"}


def dhash(img, n=8):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n + 1, n), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def audit(name, root, splits, layout):
    out = {"splits": {}, "issues": collections.Counter(), "examples": collections.defaultdict(list)}
    hashes = {}
    sizes = collections.Counter()
    box_px = collections.defaultdict(list)
    for s in splits:
        idir = root / layout.replace("split", s)
        ldir = Path(str(idir).replace("images", "labels"))
        files = sorted(p for p in idir.iterdir() if p.suffix.lower() in EXT)
        cls = collections.Counter(); n_empty = 0
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                out["issues"]["unreadable"] += 1; out["examples"]["unreadable"].append(str(p)); continue
            h, w = img.shape[:2]; sizes[(w, h)] += 1
            hashes[f"{s}/{p.name}"] = dhash(img)
            lp = ldir / (p.stem + ".txt")
            if not lp.exists():
                out["issues"]["missing_label"] += 1; continue
            lines = [l.split() for l in lp.read_text().splitlines() if l.strip()]
            if not lines: n_empty += 1
            for l in lines:
                if len(l) != 5:
                    out["issues"]["malformed_line(len!=5)"] += 1
                    if len(out["examples"]["malformed"]) < 5: out["examples"]["malformed"].append(f"{lp}: {' '.join(l)[:80]}")
                    continue
                c = l[0]; x, y, bw, bh = map(float, l[1:])
                cls[c] += 1
                if bw <= 0 or bh <= 0: out["issues"]["zero_area"] += 1
                if x - bw / 2 < -0.01 or y - bh / 2 < -0.01 or x + bw / 2 > 1.01 or y + bh / 2 > 1.01:
                    out["issues"]["out_of_bounds"] += 1
                pw, ph = bw * w, bh * h
                if min(pw, ph) < 4: out["issues"]["tiny(<4px)"] += 1
                box_px[c].append(float(np.sqrt(max(pw * ph, 0))))
        out["splits"][s] = {"images": len(files), "empty_labels": n_empty, "boxes": dict(sorted(cls.items()))}
    # exact / near duplicate analysis (dHash hamming <= 2)
    keys = list(hashes); hv = np.array([hashes[k] for k in keys], dtype=np.uint64)
    exact = collections.defaultdict(list)
    for k in keys: exact[hashes[k]].append(k)
    groups = [g for g in exact.values() if len(g) > 1]
    cross = sum(1 for g in groups if len({x.split("/")[0] for x in g}) > 1)
    near_cross = 0; near_pairs = 0
    bits = np.unpackbits(hv.view(np.uint8).reshape(-1, 8), axis=1)
    split_of = np.array([k.split("/")[0] for k in keys])
    for i in range(len(keys)):
        d = (bits[i + 1:] != bits[i]).sum(1)
        m = np.where(d <= 4)[0]
        near_pairs += len(m)
        near_cross += int((split_of[i + 1:][m] != split_of[i]).sum())
    # per-split images that have a near-dup in another split
    out["dup"] = {"exact_dup_groups": len(groups), "exact_groups_spanning_splits": cross,
                  "near_dup_pairs(hamming<=4)": near_pairs, "near_dup_pairs_across_splits": near_cross}
    out["resolutions"] = {f"{w}x{h}": n for (w, h), n in sizes.most_common(8)}
    out["box_sqrt_area_px"] = {c: {"n": len(v), "p10": round(float(np.percentile(v, 10)), 1), "median": round(float(np.median(v)), 1),
                                   "p90": round(float(np.percentile(v, 90)), 1)} for c, v in sorted(box_px.items()) if v}
    out["issues"] = dict(out["issues"]); out["examples"] = dict(out["examples"])
    return out


if __name__ == "__main__":
    res = {}
    for n, (r, sp, lay) in SOURCES.items():
        print(f"auditing {n} ...", flush=True)
        res[n] = audit(n, r, sp, lay)
        print(json.dumps(res[n], indent=1))
    (Path(__file__).parent / "source_audit.json").write_text(json.dumps(res, indent=1))
