#!/usr/bin/env python3
"""
Indian signal-head crops (from red_light/observe.py runs) for testing / fine-tuning the light-state model.

    python signal_training/head_sheets.py make [--heads 4] [--frames 4]   # sheets + crops/ + index
    python signal_training/head_sheets.py label <sheet> "<r/y/g/n/u per tile>"

Tiles are chosen per clip from the heads with most detections (and all colour-found heads), frames spread over
the shot. Labels -> signal_training/head_labels.json {key: label}; crops saved in signal_training/india_crops/.
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "red_light/runs"
SH = ROOT / "sheets"; SH.mkdir(exist_ok=True)
CR = ROOT / "india_crops"; CR.mkdir(exist_ok=True)
LAB = ROOT / "head_labels.json"
mode = sys.argv[1]
if mode == "make":
    nh = int(sys.argv[sys.argv.index("--heads") + 1]) if "--heads" in sys.argv else 4
    nf = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 4
    items = []
    for run in sorted(RUNS.iterdir()):
        if not (run / "heads.npz").exists():
            continue
        obs = json.loads((run / "obs.json").read_text())
        npz = np.load(run / "heads.npz")
        hs = sorted(obs["heads"].items(), key=lambda kv: (not kv[1].get("source", "").startswith("colour"), -kv[1]["n_det"]))
        hs = [(h, v) for h, v in hs if (v["n_det"] >= 2 or v.get("source", "").startswith("colour"))][:nh]
        for hid, v in hs:
            k = hid.replace(":", "_")
            if k not in npz.files:
                continue
            crops = npz[k]
            for i in np.linspace(0, len(crops) - 1, nf).astype(int):
                key = f"{run.name}:{hid}:{i}"
                fn = f"{run.name}_{k}_{i}.jpg"
                cv2.imwrite(str(CR / fn), crops[i])
                items.append((key, fn))
    (ROOT / "india_index.json").write_text(json.dumps(items))
    for s in range(0, len(items), 48):
        part = items[s:s + 48]
        tiles = []
        for j, (key, fn) in enumerate(part):
            im = cv2.resize(cv2.imread(str(CR / fn)), (120, 120), interpolation=cv2.INTER_NEAREST)
            c = np.zeros((136, 120, 3), np.uint8); c[:120] = im
            cv2.putText(c, f"{j + 1} {key.split(':')[0][-2:]}", (2, 132), 0, 0.4, (0, 255, 255), 1)
            tiles.append(c)
        while len(tiles) % 12:
            tiles.append(np.zeros_like(tiles[0]))
        img = np.vstack([np.hstack(tiles[i:i + 12]) for i in range(0, len(tiles), 12)])
        name = f"heads_{s // 48:02d}"
        cv2.imwrite(str(SH / f"{name}.jpg"), img); (SH / f"{name}.json").write_text(json.dumps([k for k, _ in part]))
        print(SH / f"{name}.jpg", len(part))
elif mode == "label":
    sid, labs = sys.argv[2], sys.argv[3].replace(" ", "")
    keys = json.loads((SH / f"{sid}.json").read_text())
    assert len(labs) == len(keys), f"{len(labs)} labels for {len(keys)} tiles"
    L = json.loads(LAB.read_text()) if LAB.exists() else {}
    for k, c in zip(keys, labs):
        L[k] = {"r": "red", "y": "yellow", "g": "green", "n": "not_light", "u": "unsure"}[c]
    LAB.write_text(json.dumps(L, indent=0))
    print(sid, {c: labs.count(c) for c in set(labs)}, "total", len(L))
