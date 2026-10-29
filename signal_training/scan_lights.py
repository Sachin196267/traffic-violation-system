#!/usr/bin/env python3
"""
Find traffic lights in videos (IDD model, class 'traffic light') and save crops for labelling.

    python signal_training/scan_lights.py [--every 2] [--max-frames 40] video ...

Writes signal_training/scan/<video>/crops/*.jpg and scan/<video>/lights.json
({frame_t: [[x1,y1,x2,y2,conf], ...]}); prints a one-line count per video.
"""
import argparse, json
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
ap.add_argument("--max-frames", type=int, default=40)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--weights", default=str(ROOT.parent / "runs/idd_final/weights/best.pt"))
ap.add_argument("--tag", default="", help="suffix for the scan folder (e.g. _coco)")
args = ap.parse_args()

det = YOLO(args.weights)
TL = next(k for k, v in det.names.items() if v == "traffic light")
for vp in map(Path, args.videos):
    out = ROOT / "scan" / (vp.stem + args.tag)
    if (out / "lights.json").exists():
        print(vp.stem, "already scanned"); continue
    (out / "crops").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(args.every * fps)))
    idx = list(range(0, n, step))[: args.max_frames]
    found, ncrops = {}, 0
    for fi in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            break
        r = det.predict(fr, imgsz=args.imgsz, conf=args.conf, classes=[TL], verbose=False)[0]
        boxes = [[round(float(v), 1) for v in b] + [round(float(c), 3)] for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        if boxes:
            t = round(fi / fps, 2)
            found[t] = boxes
            H, W = fr.shape[:2]
            for k, (x1, y1, x2, y2, c) in enumerate(boxes):
                pw, ph = (x2 - x1) * 0.3, (y2 - y1) * 0.3
                crop = fr[max(0, int(y1 - ph)):min(H, int(y2 + ph)), max(0, int(x1 - pw)):min(W, int(x2 + pw))]
                if crop.size:
                    cv2.imwrite(str(out / "crops" / f"{t:08.2f}_{k}.jpg"), crop); ncrops += 1
    (out / "lights.json").write_text(json.dumps({"fps": fps, "frames_sampled": len(idx), "lights": found}))
    sizes = [b[3] - b[1] for bs in found.values() for b in bs]
    print(f"{vp.stem}: {len(found)}/{len(idx)} frames with lights, {ncrops} crops, "
          f"median height {sorted(sizes)[len(sizes)//2] if sizes else 0:.0f}px", flush=True)
