"""Rate candidate videos for plate work: 1 frame/s through the plate detector, count plates and their pixel width.
Readable ~ plate >= 80 px wide (the reader's comfortable size). Writes a contact sheet of the largest crops per video.
Usage: .venv/bin/python plates/score_videos.py datasets/plate_videos"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

src = Path(sys.argv[1]); out = src / "_score"; out.mkdir(exist_ok=True)
det = YOLO("models/license_plate_best.pt")
rows = []
for v in sorted(src.glob("*.mp4")):
    cap = cv2.VideoCapture(str(v)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    widths, crops, frames = [], [], 0
    for i in range(0, n, int(round(fps))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if not ok:
            break
        frames += 1
        for b in det(f, imgsz=1280, conf=0.35, verbose=False)[0].boxes.xyxy.cpu().numpy().astype(int):
            w = b[2] - b[0]; widths.append(int(w))
            crops.append((w, f[b[1]:b[3], b[0]:b[2]]))
    cap.release()
    W = np.array(widths) if widths else np.zeros(0)
    r = {"video": v.stem, "height": h, "frames": frames, "plates": len(W),
         "per_frame": round(len(W) / max(frames, 1), 2), "readable_80px": int((W >= 80).sum()),
         "median_w": int(np.median(W)) if len(W) else 0}
    rows.append(r); print(r, flush=True)
    top = [cv2.resize(c, (240, 60)) for _, c in sorted(crops, key=lambda x: -x[0])[:24] if c.size]
    if top:
        top += [np.zeros_like(top[0])] * (-len(top) % 4)
        cv2.imwrite(str(out / f"{v.stem}.jpg"), np.vstack([np.hstack(top[i:i + 4]) for i in range(0, len(top), 4)]))
(out / "scores.json").write_text(json.dumps(rows, indent=1))
