#!/usr/bin/env python3
"""
Review sheets for labelling (numbered tiles) and recording the labels.

    python ambulance_training/sheets.py video <name> [--min 0.25]      # one tile per vehicle track (best crop)
    python ambulance_training/sheets.py web --lo 0.5 --hi 1.01         # CLIP-sorted web images in a score band
    python ambulance_training/sheets.py label <sheet-id> "<labels>"    # a=ambulance n=not u=unsure x=junk

Sheets: ambulance_training/sheets/<id>.jpg + <id>.json (tile order). Labels: ambulance_training/labels.json
{key: label}; key = "vid:<video>:<track>" or "web:<relative path>".
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SH = ROOT / "sheets"; SH.mkdir(exist_ok=True)
LAB = ROOT / "labels.json"
N_PER = 40


def tile(path, text, size=150):
    im = cv2.imread(str(path))
    c = np.zeros((size + 18, size, 3), np.uint8)
    if im is not None:
        h, w = im.shape[:2]; s = size / max(h, w)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        c[:im.shape[0], :im.shape[1]] = im
    cv2.putText(c, text, (2, size + 14), 0, 0.45, (0, 255, 255), 1)
    return c


def write(sid, items):
    """items: [(key, image path, caption)] -> sheets sid_00, sid_01..."""
    out = []
    for s in range(0, len(items), N_PER):
        part = items[s:s + N_PER]
        tiles = [tile(p, f"{i + 1} {cap}") for i, (k, p, cap) in enumerate(part)]
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        img = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
        name = f"{sid}_{s // N_PER:02d}"
        cv2.imwrite(str(SH / f"{name}.jpg"), img)
        (SH / f"{name}.json").write_text(json.dumps([k for k, _, _ in part]))
        out.append(name)
    print("\n".join(str(SH / f"{n}.jpg") + f"  ({len(items[i * N_PER:(i + 1) * N_PER])} tiles)" for i, n in enumerate(out)))


mode = sys.argv[1]
if mode == "video":
    v = sys.argv[2]; lo = float(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 0.25
    d = ROOT / "crops" / v
    j = json.loads((d / "clip.json").read_text()); m = j["meta"]
    best = {}
    for f in j["files"]:
        t = m[f]["track"]; p = m[f]["p"]["ambulance"]
        if t not in best or p > best[t][1]:
            best[t] = (f, p)
    items = [(f"vid:{v}:{t}", d / f, f"{p:.2f}") for t, (f, p) in sorted(best.items(), key=lambda x: -x[1][1]) if p >= lo]
    write(f"vid_{v}", items)
elif mode == "videos":   # many videos in one set of sheets: sheets.py videos <min> v1 v2 ...
    lo = float(sys.argv[2]); items = []
    for v in sys.argv[3:]:
        d = ROOT / "crops" / v
        j = json.loads((d / "clip.json").read_text()); m = j["meta"]
        best = {}
        for f in j["files"]:
            t = m[f]["track"]; p = m[f]["p"]["ambulance"]
            if t not in best or p > best[t][1]:
                best[t] = (f, p)
        items += [(f"vid:{v}:{t}", d / f, f"{v[-2:]} {p:.2f}") for t, (f, p) in sorted(best.items(), key=lambda x: -x[1][1]) if p >= lo]
    write("vids", items)
elif mode == "web":
    import csv
    lo = float(sys.argv[sys.argv.index("--lo") + 1]); hi = float(sys.argv[sys.argv.index("--hi") + 1])
    rows = list(csv.DictReader(open(ROOT / "clip_web/clip_scores.csv")))
    base = ROOT.parent / "datasets/external"
    items = []
    for r in sorted(rows, key=lambda r: -float(r["ambulance"])):
        p = float(r["ambulance"])
        if lo <= p < hi:
            items.append((f"web:{r['path']}", base / r["local"], f"{p:.2f}"))
    write(f"web_{lo:.2f}_{hi:.2f}", items)
elif mode == "label":
    sid, labs = sys.argv[2], sys.argv[3].replace(" ", "")
    keys = json.loads((SH / f"{sid}.json").read_text())
    assert len(labs) == len(keys), f"{len(labs)} labels for {len(keys)} tiles"
    L = json.loads(LAB.read_text()) if LAB.exists() else {}
    for k, c in zip(keys, labs):
        L[k] = {"a": "ambulance", "n": "not", "u": "unsure", "x": "junk", "p": "police", "f": "fire"}[c]
    LAB.write_text(json.dumps(L, indent=0))
    print(sid, "->", {c: labs.count(c) for c in set(labs)}, "| total labels", len(L))
