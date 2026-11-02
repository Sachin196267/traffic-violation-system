#!/usr/bin/env python3
"""
Four-wheeler crops from videos, grouped per tracked vehicle, scored with CLIP for review/labelling.

    python ambulance_training/extract_crops.py [--fps 3] [--per-track 6] video ...

Writes ambulance_training/crops/<video>/<track>_<k>.jpg and ambulance_training/crops/<video>/clip.json
({file: {"track", "t", "h", "p": {class: prob}}}) plus clip_emb.npy (ViT-B-16 embeddings, same order).
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ambulance_training"))
from clip_features import Clip  # noqa: E402

FOUR_WHEEL = {"car", "bus", "truck", "caravan", "trailer", "vehicle fallback"}
ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--fps", type=float, default=3.0)
ap.add_argument("--per-track", type=int, default=6)
ap.add_argument("--min-h", type=int, default=40)
ap.add_argument("--max-sec", type=float, default=240)
args = ap.parse_args()

det = YOLO(str(ROOT / "runs/idd_final/weights/best.pt"))
keep = [k for k, v in det.names.items() if v in FOUR_WHEEL]
clip = Clip()
for vp in map(Path, args.videos):
    out = ROOT / "ambulance_training/crops" / vp.stem
    if (out / "clip.json").exists():
        print(vp.stem, "done already"); continue
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    step = max(1, round(fps / args.fps)); last = int(args.max_sec * fps)
    imgsz = min(960, int(np.ceil(max(W, H) / 32) * 32))
    counts, last_t, meta, fi = {}, {}, {}, 0
    while fi < last and cap.grab():
        if fi % step:
            fi += 1; continue
        ok, fr = cap.retrieve()
        if not ok:
            break
        t = fi / fps
        r = det.track(fr, imgsz=imgsz, conf=0.3, classes=keep, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes.id is not None:
            for b, i, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
                x1, y1, x2, y2 = map(int, b)
                if y2 - y1 < args.min_h or counts.get(i, 0) >= args.per_track or t - last_t.get(i, -9) < 1.0:
                    continue
                pw, ph = int(0.05 * (x2 - x1)), int(0.05 * (y2 - y1))
                crop = fr[max(0, y1 - ph):min(H, y2 + ph), max(0, x1 - pw):min(W, x2 + pw)]
                fn = f"{i}_{counts.get(i, 0)}.jpg"
                cv2.imwrite(str(out / fn), crop)
                meta[fn] = {"track": i, "t": round(t, 2), "h": y2 - y1, "cls": det.names[c]}
                counts[i] = counts.get(i, 0) + 1; last_t[i] = t
        fi += 1
    files = sorted(meta)
    P, E = clip.score([str(out / f) for f in files])
    for f, p in zip(files, P):
        meta[f]["p"] = p
    np.save(out / "clip_emb.npy", E)
    (out / "clip.json").write_text(json.dumps({"files": files, "meta": meta}))
    top = sorted(files, key=lambda f: -meta[f]["p"]["ambulance"])[:5]
    print(f"{vp.stem}: {len(files)} crops from {len(counts)} vehicles; top ambulance scores "
          f"{[round(meta[f]['p']['ambulance'], 2) for f in top]}", flush=True)
