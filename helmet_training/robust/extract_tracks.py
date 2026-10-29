#!/usr/bin/env python3
"""
Stage 1 of the robust helmet pipeline: tracked two-wheeler riders -> head crops for labelling.

Per video (sampled at --fps):
  1. IDD model (runs/idd_final) + ByteTrack follows two-wheelers / riders
  2. every --every-k sampled frames, each track big enough gets an expanded crop (upscaled if small)
  3. heads inside the crop = pose-keypoint heads, replaced by tighter helmet-model boxes when they overlap
  4. each head is saved as a context patch (head box x2.0, square, 128 px) + metadata

Output: helmet_training/robust/data/<video>/{patches/*.jpg, heads.json}
heads.json rows: {id, video, frame, t, track, slot, box (frame coords), head_px, patch}
"slot" is the head's order left->right inside the vehicle box, so the same person keeps the
same (track, slot) across frames for multi-frame voting.
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "helmet_training/robust/data"
TW_CLASSES = ("motorcycle", "bicycle", "rider")
PATCH = 128

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--fps", type=float, default=5.0, help="tracker sampling rate")
ap.add_argument("--every-k", type=int, default=5, help="save heads every k sampled frames per track")
ap.add_argument("--max-sec", type=float, default=240)
ap.add_argument("--min-tw", type=int, default=40, help="min two-wheeler box height (px)")
ap.add_argument("--min-head", type=int, default=8, help="min head box side (px, frame scale)")
args = ap.parse_args()

idd = YOLO(str(ROOT / "runs/idd_final/weights/best.pt"))
pose = YOLO(str(ROOT / "yolo11s-pose.pt"))
helm = YOLO(str(ROOT / "runs/helmet_nohelmet_v2_final/weights/best.pt"))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def upscale(crop):
    h0, w0 = crop.shape[:2]
    s = max(1.0, 384 / min(h0, w0))
    return (cv2.resize(crop, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_CUBIC) if s > 1 else crop), s


def heads_batch(crops):
    """Head boxes (crop coords) per crop: pose heads, swapped for tighter helmet-model boxes where they overlap."""
    ups = [upscale(c) for c in crops]
    bigs = [u[0] for u in ups]
    out = []
    for i in range(0, len(bigs), 16):
        chunk = bigs[i:i + 16]
        prs = pose.predict(chunk, imgsz=640, conf=0.25, verbose=False)
        hrs = helm.predict(chunk, imgsz=640, conf=0.2, verbose=False)
        for r, hr in zip(prs, hrs):
            heads = []
            if r.keypoints is not None:
                for pb, kp in zip(r.boxes.xyxy.tolist(), r.keypoints.data.tolist()):
                    vis = [k for k in kp[:5] if k[2] > 0.3]
                    sh = [kp[5], kp[6]] if kp[5][2] > 0.3 and kp[6][2] > 0.3 else None
                    sw = abs(sh[0][0] - sh[1][0]) if sh else 0
                    ears = abs(kp[3][0] - kp[4][0]) if kp[3][2] > 0.3 and kp[4][2] > 0.3 else 0
                    size = max(0.8 * sw, 1.4 * ears, 0.35 * (pb[2] - pb[0]))
                    cx = np.mean([k[0] for k in vis]) if vis else ((sh[0][0] + sh[1][0]) / 2 if sh else (pb[0] + pb[2]) / 2)
                    heads.append([cx - size / 2, pb[1], cx + size / 2, pb[1] + 1.1 * size])
            for hb in hr.boxes.xyxy.tolist():
                m = [j for j, q in enumerate(heads) if iou(q, hb) > 0.25]
                if m:
                    heads[m[0]] = hb
                else:
                    heads.append(hb)
            keep = []
            for h in sorted(heads, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
                if all(iou(h, k) < 0.4 for k in keep):
                    keep.append(h)
            out.append(keep)
    return [[[v / s for v in h] for h in hs] for hs, (_, s) in zip(out, ups)]


def patch_of(frame, hb):
    """Square context patch around a head (2x head size), resized to PATCH."""
    H, W = frame.shape[:2]
    cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
    half = max(hb[2] - hb[0], hb[3] - hb[1])
    x1, y1, x2, y2 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    pad = [max(0, -y1), max(0, y2 - H), max(0, -x1), max(0, x2 - W)]
    p = frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    p = cv2.copyMakeBorder(p, *pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(p, (PATCH, PATCH), interpolation=cv2.INTER_CUBIC)


for vp in args.videos:
    vp = Path(vp)
    if (OUT / vp.stem / "heads.json").exists():
        print(vp.stem, "done already"); continue
    idd.predictor = None  # fresh tracker per video
    tag = vp.stem
    od = OUT / tag
    (od / "patches").mkdir(parents=True, exist_ok=True)
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
        jobs = []
        for b, tid, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
            if r.names[c] not in TW_CLASSES or b[3] - b[1] < args.min_tw:
                continue
            k = seen.get(tid, 0)
            seen[tid] = k + 1
            if k % args.every_k:
                continue
            bw, bh = b[2] - b[0], b[3] - b[1]
            up = 0.6 if r.names[c] != "rider" else 0.25
            x1, y1, x2, y2 = map(int, [max(0, b[0] - 0.2 * bw), max(0, b[1] - up * bh), min(W, b[2] + 0.2 * bw), min(H, b[3] + 0.05 * bh)])
            if x2 - x1 > 4 and y2 - y1 > 4:
                jobs.append((b, tid, c, x1, y1, f[y1:y2, x1:x2]))
        if not jobs:
            continue
        for (b, tid, c, x1, y1, _), hs in zip(jobs, heads_batch([j[5] for j in jobs])):
            bw, bh = b[2] - b[0], b[3] - b[1]
            hs = [[h[0] + x1, h[1] + y1, h[2] + x1, h[3] + y1] for h in hs]
            # keep heads in the upper part of the vehicle/rider region
            hs = [h for h in hs if min(h[2] - h[0], h[3] - h[1]) >= args.min_head and (h[1] + h[3]) / 2 < b[1] + 0.6 * bh]
            hs.sort(key=lambda h: h[0] + h[2])
            for slot, h in enumerate(hs):
                hid = len(rows)
                name = f"{tag}_{hid:05d}.jpg"
                cv2.imwrite(str(od / "patches" / name), patch_of(f, h), [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"id": hid, "video": tag, "frame": fi, "t": round(fi / fps, 2), "track": tid, "slot": slot,
                             "rel_x": round(((h[0] + h[2]) / 2 - b[0]) / bw, 3), "vcls": r.names[c],
                             "box": [round(v, 1) for v in h], "head_px": round(min(h[2] - h[0], h[3] - h[1]), 1),
                             "frame_hw": [H, W], "patch": name})
    (od / "heads.json").write_text(json.dumps(rows))
    print(f"{tag}: {len(rows)} heads, {len({r['track'] for r in rows})} tracks", flush=True)
