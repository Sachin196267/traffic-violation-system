#!/usr/bin/env python3
"""
Stage 1: detect + track vehicles, with camera stabilisation.

  python speed_estimation/track.py <video> [--fps 10] [--start 0] [--max-sec 120] [--detector coco|idd]

Camera shake / pans: every frame is registered to the first one (homography from background features, vehicles
masked out) and boxes are stored in the first frame's coordinates. Registration failure (scene cut, big pan) ends
the analysed segment.

Writes <out>/<video>/tracks.json:
  {video, fps_video, W, H, step, t0, t_end, stop_reason, max_shift_px, tracks: {id: {cls, t, box (first-frame coords), box_img (frame coords), conf}}}
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DETECTORS = {  # weights, {class name in model: our class}
    "coco": (ROOT / "yolo11s.pt", {"car": "car", "motorcycle": "motorcycle", "bus": "bus", "truck": "truck"}),
    "idd": (ROOT / "runs/idd_final/weights/best.pt",
            {"car": "car", "motorcycle": "motorcycle", "bus": "bus", "truck": "truck", "autorickshaw": "autorickshaw"}),
}

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default=str(ROOT / "speed_estimation/runs"))
ap.add_argument("--fps", type=float, default=10.0, help="processing rate (frames per second of video)")
ap.add_argument("--start", type=float, default=0.0, help="start time (s)")
ap.add_argument("--max-sec", type=float, default=0.0, help="length to process (0 = to the end)")
ap.add_argument("--detector", choices=DETECTORS, default="coco")
ap.add_argument("--imgsz", type=int, default=1280)
args = ap.parse_args()

vp = Path(args.video)
cap = cv2.VideoCapture(str(vp))
if not cap.isOpened():
    raise SystemExit(f"cannot read video: {vp}")
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W, H = int(cap.get(3)), int(cap.get(4))
step = max(1, round(fps / args.fps))
f0 = int(args.start * fps)
f1 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.max_sec:
    f1 = min(f1, f0 + int(args.max_sec * fps))
cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

weights, keep = DETECTORS[args.detector]
det = YOLO(str(weights))
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
SC = 960 / W  # registration works on a 960-px-wide copy
tracks, ref, shifts, stop_reason, t_last = {}, None, [], "end of requested segment", args.start
fi = f0 - 1
while True:
    ok = cap.grab()
    fi += 1
    if not ok or fi >= f1:
        break
    if (fi - f0) % step:
        continue
    ok, frame = cap.retrieve()
    if not ok:
        break
    t = fi / fps
    r = det.track(frame, imgsz=args.imgsz, conf=0.25, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    # --- register to the first frame (background only) ---
    grey = cv2.cvtColor(cv2.resize(frame, (960, round(H * SC))), cv2.COLOR_BGR2GRAY)
    mask = np.full(grey.shape, 255, np.uint8)
    for b in r.boxes.xyxy.tolist():
        x1, y1, x2, y2 = [int(v * SC) for v in b]
        mask[max(0, y1):y2, max(0, x1):x2] = 0
    kp, des = orb.detectAndCompute(grey, mask)
    if ref is None:
        ref = (kp, des)
        Hm = np.eye(3)
    else:
        Hm = None
        if des is not None and len(kp) > 50:
            good = [m for m, n in (p for p in bf.knnMatch(des, ref[1], k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
            if len(good) > 40:
                src = np.float32([kp[m.queryIdx].pt for m in good]); dst = np.float32([ref[0][m.trainIdx].pt for m in good])
                Hs, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
                if Hs is not None and inl.sum() > 30:
                    S = np.diag([SC, SC, 1.0])
                    Hm = np.linalg.inv(S) @ Hs @ S          # full-resolution homography frame -> first frame
        if Hm is None:
            stop_reason = f"camera view changed (scene cut or large pan) at {t:.1f}s"
            break
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    moved = cv2.perspectiveTransform(corners[None], Hm)[0]
    shifts.append((round(t, 3), round(float(np.abs(moved - corners).max()), 1)))
    t_last = t
    if r.boxes.id is None:
        continue
    for b, tid, c, cf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
        name = keep.get(r.names[c])
        if name is None:
            continue
        x1, y1, x2, y2 = b
        q = cv2.perspectiveTransform(np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [(x1 + x2) / 2, y2]]], float), Hm)[0]
        bw = np.linalg.norm(q[1] - q[0]); bh = np.linalg.norm(q[3] - q[0])
        bx, by = q[4]                                       # bottom centre in first-frame coordinates
        tr = tracks.setdefault(str(tid), {"cls": [], "t": [], "box": [], "box_img": [], "conf": []})
        tr["cls"].append(name)
        tr["t"].append(round(t, 4))
        tr["box"].append([round(v, 1) for v in (bx - bw / 2, by - bh, bx + bw / 2, by)])
        tr["box_img"].append([round(v, 1) for v in b])     # as seen in the frame (for drawing)
        tr["conf"].append(round(cf, 3))

for tr in tracks.values():  # majority class per track
    tr["cls"] = max(set(tr["cls"]), key=tr["cls"].count)
out = Path(args.out) / vp.stem
out.mkdir(parents=True, exist_ok=True)
max_shift = max((m for _, m in shifts), default=0.0)
json.dump({"video": str(vp), "fps_video": fps, "W": W, "H": H, "step": step, "t0": args.start, "t_end": round(t_last, 2),
           "stop_reason": stop_reason, "detector": args.detector, "max_shift_px": max_shift, "shifts": shifts,
           "tracks": tracks}, open(out / "tracks.json", "w"))
print(json.dumps({"video": vp.name, "tracks": len(tracks), "frames": len(shifts), "analysed_s": [args.start, round(t_last, 2)],
                  "stop_reason": stop_reason, "max_camera_shift_px": max_shift}))
