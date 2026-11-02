#!/usr/bin/env python3
"""
Step 2 - "heads on the seat" evidence for every unit crop.

Heads are found on the (upscaled) crop by the pose model (nose/eyes/ears keypoints) and the
head/helmet detector, merged, then kept only if they plausibly belong to a person SITTING ON
THIS motorcycle:
    * head centre x inside the bike box widened by 10 %           (not the rider on the next bike)
    * head centre y between  bike_top - 1.6 * bike_h  and  bike_top + 0.35 * bike_h
    * head width between 0.10 and 0.55 * bike width                 (a person further back is smaller)
Writes data/<video>/heads_seat.json: {crop: {"n": heads on seat, "sizes": [...]}}.
    python triple_training/heads_seat.py [video ...]      (default: all videos with units.json)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
from focus import letterbox_geom  # noqa: E402

pose = YOLO(str(R.parent / "yolo11s-pose.pt"))
helm = YOLO(str(R.parent / "runs/helmet_nohelmet_v2_final/weights/best.pt"))
UP = 2.0                                             # 256 -> 512 before detection


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def heads_in(imgs):
    ups = [cv2.resize(im, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC) for im in imgs]
    out = []
    for r, hr in zip(pose.predict(ups, imgsz=640, conf=0.2, verbose=False),
                     helm.predict(ups, imgsz=640, conf=0.2, verbose=False)):
        heads = []
        if r.keypoints is not None:
            for pb, kp in zip(r.boxes.xyxy.tolist(), r.keypoints.data.tolist()):
                vis = [k for k in kp[:5] if k[2] > 0.3]
                if not vis:
                    continue
                cx, cy = np.mean([k[0] for k in vis]), np.mean([k[1] for k in vis])
                ears = abs(kp[3][0] - kp[4][0]) if kp[3][2] > 0.3 and kp[4][2] > 0.3 else 0
                sz = max(1.3 * ears, 0.28 * (pb[2] - pb[0]), 6)
                heads.append([cx - sz / 2, cy - sz / 2, cx + sz / 2, cy + sz / 2])
        for hb in hr.boxes.xyxy.tolist():
            m = [j for j, q in enumerate(heads) if iou(q, hb) > 0.2]
            if m:
                heads[m[0]] = hb
            else:
                heads.append(hb)
        keep = []
        for h in sorted(heads, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
            if all(iou(h, k) < 0.3 for k in keep):
                keep.append([v / UP for v in h])
        out.append(keep)
    return out


def on_seat(heads, moto_lb):
    mx1, my1, mx2, my2 = moto_lb
    mw, mh = mx2 - mx1, my2 - my1
    kept = []
    for h in heads:
        cx, cy, hw = (h[0] + h[2]) / 2, (h[1] + h[3]) / 2, h[2] - h[0]
        if (mx1 - 0.10 * mw <= cx <= mx2 + 0.10 * mw and my1 - 1.6 * mh <= cy <= my1 + 0.35 * mh
                and 0.10 * mw <= hw <= 0.55 * mw):
            kept.append(round(hw / mw, 3))
    return kept


def moto_in_letterbox(u):
    x1, y1, x2, y2 = u["box"]
    s, ox, oy = letterbox_geom(x2 - x1, y2 - y1)
    m = u["moto"]
    return [(m[0] - x1) * s + ox, (m[1] - y1) * s + oy, (m[2] - x1) * s + ox, (m[3] - y1) * s + oy]


def run(video):
    f = R / "data" / video / "units.json"
    out_f = R / "data" / video / "heads_seat.json"
    if out_f.exists():
        return
    U = [u for u in json.loads(f.read_text()) if "moto" in u]
    res = {}
    for i in range(0, len(U), 32):
        chunk = U[i:i + 32]
        imgs = [cv2.imread(str(R / "data" / video / "crops" / u["crop"])) for u in chunk]
        for u, hs in zip(chunk, heads_in(imgs)):
            sizes = on_seat(hs, moto_in_letterbox(u))
            res[u["crop"]] = {"n": len(sizes), "sizes": sizes, "all": len(hs)}
    out_f.write_text(json.dumps(res))
    print(video, len(res), "crops", flush=True)


if __name__ == "__main__":
    vids = sys.argv[1:] or sorted(p.parent.name for p in (R / "data").glob("*/units.json"))
    for v in vids:
        run(v)
