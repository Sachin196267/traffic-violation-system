#!/usr/bin/env python3
"""
Stage 1 - two-wheeler "units" for triple-riding labelling and training.

A unit = one tracked motorcycle/scooter + every rider sitting on it.  Per video (sampled at
--fps) the IDD detector + ByteTrack follows motorcycles; rider/person boxes standing on a
motorcycle are attached to it; every --every-k sampled frames the union box is cropped and
saved (letterboxed to 256 px, upscaled if small) together with cheap evidence:
    n_rider_boxes  rider/person boxes attached to the motorcycle
    n_heads        heads found by pose + head model inside the crop (upper part only)

Output: triple_training/data/<video>/{crops/*.jpg, units.json}
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "triple_training/data"
SIZE = 256

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--fps", type=float, default=5.0)
ap.add_argument("--every-k", type=int, default=3, help="save a crop every k sampled frames per track")
ap.add_argument("--max-sec", type=float, default=240)
ap.add_argument("--min-h", type=int, default=40, help="min motorcycle box height in px")
args = ap.parse_args()

idd = YOLO(str(ROOT / "runs/idd_final/weights/best.pt"))
pose = YOLO(str(ROOT / "yolo11s-pose.pt"))
helm = YOLO(str(ROOT / "runs/helmet_nohelmet_v2_final/weights/best.pt"))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def on_bike(r, m):
    """Is rider/person box r sitting on motorcycle box m?"""
    mw, mh = m[2] - m[0], m[3] - m[1]
    cx = (r[0] + r[2]) / 2
    return (m[0] - 0.15 * mw <= cx <= m[2] + 0.15 * mw          # above the bike horizontally
            and r[1] < m[1] + 0.3 * mh                           # starts above / at the bike top
            and m[1] - 0.2 * mh <= r[3] <= m[3] + 0.25 * mh)     # ends on the bike


def letterbox(img, size=SIZE):
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                   interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    out = np.full((size, size, 3), 114, np.uint8)
    y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def head_counts(crops):
    """Number of distinct heads per crop (pose heads + head-model boxes), on upscaled crops."""
    ups = []
    for c in crops:
        h0, w0 = c.shape[:2]
        s = max(1.0, 384 / max(1, min(h0, w0)))
        ups.append(cv2.resize(c, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_CUBIC) if s > 1 else c)
    out = []
    for i in range(0, len(ups), 16):
        chunk = ups[i:i + 16]
        for img, r, hr in zip(chunk, pose.predict(chunk, imgsz=640, conf=0.25, verbose=False),
                              helm.predict(chunk, imgsz=640, conf=0.2, verbose=False)):
            H = img.shape[0]
            heads = []
            if r.keypoints is not None:
                for pb, kp in zip(r.boxes.xyxy.tolist(), r.keypoints.data.tolist()):
                    vis = [k for k in kp[:5] if k[2] > 0.3]
                    if not vis:
                        continue
                    cx, cy = np.mean([k[0] for k in vis]), np.mean([k[1] for k in vis])
                    sz = max(8, 0.3 * (pb[2] - pb[0]))
                    heads.append([cx - sz / 2, cy - sz / 2, cx + sz / 2, cy + sz / 2])
            for hb in hr.boxes.xyxy.tolist():
                m = [j for j, q in enumerate(heads) if iou(q, hb) > 0.2]
                if m:
                    heads[m[0]] = hb
                else:
                    heads.append(hb)
            keep = []
            for h in sorted(heads, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
                if (h[1] + h[3]) / 2 < 0.75 * H and all(iou(h, k) < 0.3 for k in keep):
                    keep.append(h)
            out.append(len(keep))
    return out


for vp in args.videos:
    vp = Path(vp)
    tag = vp.stem
    od = OUT / tag
    if (od / "units.json").exists():
        print(tag, "done already", flush=True)
        continue
    (od / "crops").mkdir(parents=True, exist_ok=True)
    idd.predictor = None                                   # fresh tracker ids per video
    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(fps / args.fps))
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(args.max_sec * fps))
    rows, seen, fi = [], {}, -1
    while True:
        ok = cap.grab()
        fi += 1
        if not ok or fi >= n:
            break
        if fi % step:
            continue
        ok, f = cap.retrieve()
        if not ok:
            break
        H, W = f.shape[:2]
        r = idd.track(f, imgsz=960, conf=0.3, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes.id is None:
            continue
        boxes = r.boxes.xyxy.tolist()
        names = [r.names[c] for c in r.boxes.cls.int().tolist()]
        ids = r.boxes.id.int().tolist()
        people = [b for b, nm in zip(boxes, names) if nm in ("rider", "person")]
        jobs = []
        for b, nm, tid in zip(boxes, names, ids):
            if nm != "motorcycle" or b[3] - b[1] < args.min_h:
                continue
            k = seen.get(tid, 0)
            seen[tid] = k + 1
            if k % args.every_k:
                continue
            att = [p for p in people if on_bike(p, b)]
            u = [min([b[0]] + [p[0] for p in att]), min([b[1]] + [p[1] for p in att]),
                 max([b[2]] + [p[2] for p in att]), max([b[3]] + [p[3] for p in att])]
            uw, uh = u[2] - u[0], u[3] - u[1]
            if not att:                                     # no rider box: riders sit above the bike
                u[1] = u[1] - 0.9 * uh
                uh = u[3] - u[1]
            x1, y1 = int(max(0, u[0] - 0.12 * uw)), int(max(0, u[1] - 0.08 * uh))
            x2, y2 = int(min(W, u[2] + 0.12 * uw)), int(min(H, u[3] + 0.04 * uh))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            jobs.append(dict(tid=tid, moto=b, n_rider_boxes=len(att), box=[x1, y1, x2, y2],
                             crop=f[y1:y2, x1:x2].copy(),
                             edge=x1 <= 1 or y1 <= 1 or x2 >= W - 1 or y2 >= H - 1))
        if not jobs:
            continue
        for j, nh in zip(jobs, head_counts([j["crop"] for j in jobs])):
            uid = len(rows)
            name = f"{tag}_{uid:05d}.jpg"
            cv2.imwrite(str(od / "crops" / name), letterbox(j["crop"]), [cv2.IMWRITE_JPEG_QUALITY, 93])
            rows.append(dict(id=uid, video=tag, frame=fi, t=round(fi / fps, 2), track=j["tid"],
                             box=j["box"], moto=[round(v, 1) for v in j["moto"]], crop_h=j["box"][3] - j["box"][1],
                             n_rider_boxes=j["n_rider_boxes"], n_heads=nh, edge=j["edge"],
                             frame_hw=[H, W], crop=name))
    (od / "units.json").write_text(json.dumps(rows))
    print(f"{tag}: {len(rows)} crops, {len({r['track'] for r in rows})} motorcycle tracks", flush=True)
