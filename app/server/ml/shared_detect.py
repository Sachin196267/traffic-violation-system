#!/usr/bin/env python3
"""
Shared vehicle detection + tracking pass (IDD detector, ByteTrack), run ONCE per video.
Mirrors the tools' own loop exactly (same frames: step = round(fps / 5), same max-sec, same model, same
settings), so the tools get identical detections and track IDs when they replay this cache instead of
running the detector themselves (see hook/sitecustomize.py).

    python shared_detect.py <video> <cache.pkl> [--max-sec S]
"""
import argparse, pickle, sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("out")
ap.add_argument("--max-sec", type=float, default=0); ap.add_argument("--fps", type=float, default=5.0)
a = ap.parse_args()
idd = YOLO(str(ROOT / "runs/idd_final/weights/best.pt"))
cap = cv2.VideoCapture(a.video)
if not cap.isOpened() or not cap.read()[0]:
    sys.exit("cannot read video")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
step = max(1, round(fps / a.fps))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if a.max_sec:
    n = min(n, int(a.max_sec * fps))
frames = []
fi = -1
while True:
    ok = cap.grab(); fi += 1
    if not ok or fi >= n:
        break
    if fi % step:
        continue
    ok, f = cap.retrieve()
    r = idd.track(f, imgsz=960, conf=0.3, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    b = r.boxes
    frames.append(dict(fi=fi, xyxy=b.xyxy.cpu().numpy().astype(np.float32), cls=b.cls.cpu().numpy().astype(np.float32),
                       conf=b.conf.cpu().numpy().astype(np.float32), id=None if b.id is None else b.id.cpu().numpy().astype(np.float32)))
with open(a.out, "wb") as fh:
    pickle.dump(dict(video=a.video, step=step, n=n, names=idd.names, frames=frames), fh)
print(f"shared detection: {len(frames)} frames cached")
