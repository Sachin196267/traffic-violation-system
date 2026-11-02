#!/usr/bin/env python3
"""
Label-free helmet-model check across every video in videos/ (standalone; not part of the app).

For each video, sample one frame every --every seconds (max --max-samples) and record:
  * two-wheelers visible (IDD idd_traffic_v2 motorcycle/rider boxes, >= --min-h px tall)
  * helmet model on the full frame at 640 and 1280
  * helmet model on each two-wheeler crop (box extended upward to include the head)
Saves annotated rider crops (for visual review) and summary.json per video under
helmet_training/reports/all_videos/<video>/.
"""
import argparse, json, re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(ROOT / "runs/helmet_nohelmet_final/weights/best.pt"))
ap.add_argument("--every", type=float, default=2.0)
ap.add_argument("--max-samples", type=int, default=90)
ap.add_argument("--min-h", type=int, default=40)
ap.add_argument("--device", default="mps")
a = ap.parse_args()

idd = YOLO(str(ROOT / "runs/detect/runs/detect/idd_traffic_v2/weights/best.pt"))
helm = YOLO(a.weights)
CONF = {"helmet": 0.45, "no_helmet": 0.50}
OUT = ROOT / "helmet_training/reports/all_videos"


def keep(r):
    return [(r.names[int(b.cls)], float(b.conf), [int(v) for v in b.xyxy[0]])
            for b in r.boxes if float(b.conf) >= CONF[r.names[int(b.cls)]]]


videos = sorted((ROOT / "videos").glob("*.mp4")) + sorted((ROOT / "videos/test").glob("*.mp4"))
overall = {}
for vp in videos:
    tag = re.sub(r"[^A-Za-z0-9]+", "_", vp.stem)[:48].strip("_")
    od = OUT / tag
    (od / "crops").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * a.every))
    idxs = list(range(0, n, step))[: a.max_samples]
    s = dict(video=vp.name, resolution=f"{int(cap.get(3))}x{int(cap.get(4))}", samples=0, two_wheelers=0,
             two_wheeler_heights=[], full640={"helmet": 0, "no_helmet": 0}, full1280={"helmet": 0, "no_helmet": 0},
             crop={"crops": 0, "head_found": 0, "helmet": 0, "no_helmet": 0})
    saved = 0
    best_frame = (-1, None, None)
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            break
        s["samples"] += 1
        H, W = f.shape[:2]
        r = idd.predict(f, imgsz=1280, conf=0.35, device=a.device, verbose=False)[0]
        tw = [[int(v) for v in b.xyxy[0]] for b in r.boxes
              if r.names[int(b.cls)] in ("motorcycle", "rider") and (b.xyxy[0][3] - b.xyxy[0][1]) >= a.min_h]
        s["two_wheelers"] += len(tw)
        s["two_wheeler_heights"] += [y2 - y1 for _, y1, _, y2 in tw]
        for sz, key in ((640, "full640"), (1280, "full1280")):
            for c, _, _ in keep(helm.predict(f, imgsz=sz, conf=0.25, device=a.device, verbose=False)[0]):
                s[key][c] += 1
        for x1, y1, x2, y2 in tw:
            h = y2 - y1
            cx1, cy1, cx2, cy2 = max(0, x1 - h // 4), max(0, y1 - h), min(W, x2 + h // 4), y2
            crop = f[cy1:cy2, cx1:cx2]
            p = helm.predict(crop, imgsz=320, conf=0.25, device=a.device, verbose=False)[0]
            dets = keep(p)
            s["crop"]["crops"] += 1
            s["crop"]["head_found"] += bool(dets)
            for c, _, _ in dets:
                s["crop"][c] += 1
            if saved < 40 and h >= 60:
                vis = crop.copy()
                for c, cf, (bx1, by1, bx2, by2) in dets:
                    col = (0, 200, 0) if c == "helmet" else (0, 0, 255)
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), col, 2)
                    cv2.putText(vis, f"{c[:2].upper()} {cf:.2f}", (bx1, max(12, by1 - 3)), 0, 0.5, col, 1)
                if not dets:
                    cv2.putText(vis, "NONE", (4, 16), 0, 0.5, (0, 255, 255), 1)
                cv2.imwrite(str(od / "crops" / f"{fi:06d}_{x1}.jpg"), vis)
                saved += 1
        if len(tw) > best_frame[0]:
            best_frame = (len(tw), f, fi)
    if best_frame[1] is not None:
        f = best_frame[1]
        cv2.imwrite(str(od / "busiest_frame_1280.jpg"),
                    helm.predict(f, imgsz=1280, conf=0.45, device=a.device, verbose=False)[0].plot())
    hs = s.pop("two_wheeler_heights")
    s["two_wheeler_height_px_median"] = int(np.median(hs)) if hs else 0
    s["crop"]["head_found_rate"] = round(s["crop"]["head_found"] / max(1, s["crop"]["crops"]), 3)
    (od / "summary.json").write_text(json.dumps(s, indent=1))
    overall[tag] = s
    print(json.dumps(s), flush=True)
(OUT / "summary_all.json").write_text(json.dumps(overall, indent=1))
