#!/usr/bin/env python3
"""
Label-free check of the motorcycle-level helmet model (box = motorcycle + riders) on every video in videos/.
Samples one frame every --every seconds, runs the model on the full frame, and saves annotated frames +
zoomed crops of each detection for visual review.

    .venv/bin/python helmet_training/eval_moto_videos.py --weights runs/helmet_moto_kaggle/best.pt
Outputs: helmet_training/reports/moto_videos/<video>/{summary.json, frames/, crops/}
"""
import argparse, json, re
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--every", type=float, default=2.0)
ap.add_argument("--max-samples", type=int, default=90)
ap.add_argument("--device", default="mps")
a = ap.parse_args()

m = YOLO(a.weights)
OUT = ROOT / "helmet_training/reports/moto_videos"
overall = {}
for vp in sorted((ROOT / "videos").glob("*.mp4")) + sorted((ROOT / "videos/test").glob("*.mp4")):
    tag = re.sub(r"[^A-Za-z0-9]+", "_", vp.stem)[:48].strip("_")
    od = OUT / tag
    for d in ("frames", "crops"):
        (od / d).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s = {"video": vp.name, "samples": 0, "helmet": 0, "no_helmet": 0, "frames_with_dets": 0}
    frames_saved = crops_saved = 0
    for fi in list(range(0, n, max(1, int(fps * a.every))))[: a.max_samples]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            break
        s["samples"] += 1
        r = m.predict(f, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        if len(r.boxes):
            s["frames_with_dets"] += 1
        for b in r.boxes:
            c = r.names[int(b.cls)]
            s[c] += 1
            if crops_saved < 60:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                pad = (y2 - y1) // 4
                crop = f[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad].copy()
                if crop.size:
                    crop = cv2.resize(crop, (200, int(200 * crop.shape[0] / max(1, crop.shape[1]))))
                    col = (0, 200, 0) if c == "helmet" else (0, 0, 255)
                    cv2.putText(crop, f"{c} {float(b.conf):.2f}", (3, 14), 0, 0.45, col, 1)
                    cv2.imwrite(str(od / "crops" / f"{fi:06d}_{x1}_{c}.jpg"), crop)
                    crops_saved += 1
        if frames_saved < 6 and len(r.boxes) >= 2:
            cv2.imwrite(str(od / "frames" / f"{fi:06d}.jpg"), r.plot(line_width=2, font_size=12))
            frames_saved += 1
    (od / "summary.json").write_text(json.dumps(s, indent=1))
    overall[tag] = s
    print(json.dumps(s), flush=True)
(OUT / "summary_all.json").write_text(json.dumps(overall, indent=1))
