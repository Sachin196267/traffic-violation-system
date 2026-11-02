#!/usr/bin/env python3
"""
Two-stage helmet check on every video in videos/ (standalone; not part of the app):
  1. IDD detector finds motorcycles and riders; each motorcycle is merged with overlapping rider boxes.
  2. Helmet classifier labels the merged crop helmet / no_helmet.
Saves labelled crops + annotated frames for visual review.

    .venv/bin/python helmet_training/eval_two_stage_videos.py --idd <idd.pt> --cls <cls.pt>
Outputs: helmet_training/reports/two_stage_videos/<video>/{summary.json, frames/, crops/}
"""
import argparse, json, re
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--idd", required=True)
ap.add_argument("--idd-imgsz", type=int, default=1280)
ap.add_argument("--cls", required=True, help="classifier (yolo -cls) or motorcycle-level helmet detector")
ap.add_argument("--no-helmet-thr", type=float, default=0.5, help="P(no_helmet) needed to call no_helmet")
ap.add_argument("--min-h", type=int, default=40, help="ignore two-wheelers shorter than this (px)")
ap.add_argument("--every", type=float, default=2.0)
ap.add_argument("--max-samples", type=int, default=90)
ap.add_argument("--device", default="mps")
ap.add_argument("--out", default=str(ROOT / "helmet_training/reports/two_stage_videos"))
a = ap.parse_args()

idd, cls = YOLO(a.idd), YOLO(a.cls)
NH = [k for k, v in cls.names.items() if v == "no_helmet"][0]


def overlap(a_, b_):
    ix = max(0, min(a_[2], b_[2]) - max(a_[0], b_[0]))
    iy = max(0, min(a_[3], b_[3]) - max(a_[1], b_[1]))
    return ix * iy / max(1, (b_[2] - b_[0]) * (b_[3] - b_[1]))


def two_wheelers(r):
    motos = [list(map(int, b.xyxy[0])) for b in r.boxes if r.names[int(b.cls)] in ("motorcycle", "bicycle")]
    riders = [list(map(int, b.xyxy[0])) for b in r.boxes if r.names[int(b.cls)] == "rider"]
    out, used = [], set()
    for m in motos:
        box = m[:]
        for j, rd in enumerate(riders):
            if overlap(m, rd) > 0.2 or overlap(rd, m) > 0.2:
                box = [min(box[0], rd[0]), min(box[1], rd[1]), max(box[2], rd[2]), max(box[3], rd[3])]
                used.add(j)
        out.append(box)
    out += [rd for j, rd in enumerate(riders) if j not in used]  # rider without a detected bike
    return out


OUT = Path(a.out)
overall = {}
for vp in sorted((ROOT / "videos").glob("*.mp4")) + sorted((ROOT / "videos/test").glob("*.mp4")):
    tag = re.sub(r"[^A-Za-z0-9]+", "_", vp.stem)[:48].strip("_")
    od = OUT / tag
    for d in ("frames", "crops"):
        (od / d).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s = {"video": vp.name, "samples": 0, "two_wheelers": 0, "helmet": 0, "no_helmet": 0}
    frames_saved = crops_saved = 0
    for fi in list(range(0, n, max(1, int(fps * a.every))))[: a.max_samples]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            break
        s["samples"] += 1
        H, W = f.shape[:2]
        boxes = [b for b in two_wheelers(idd.predict(f, imgsz=a.idd_imgsz, conf=0.3, device=a.device, verbose=False)[0])
                 if b[3] - b[1] >= a.min_h]
        vis = f.copy()
        for x1, y1, x2, y2 in boxes:
            px, py = (x2 - x1) // 14, (y2 - y1) // 14
            crop = f[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)]
            if cls.task == "classify":
                p = cls.predict(crop, imgsz=224, device=a.device, verbose=False)[0].probs.data.tolist()
            else:  # motorcycle-level detector used as a crop classifier: take its most confident box
                rr = cls.predict(crop, imgsz=640, conf=0.15, device=a.device, verbose=False)[0]
                if not len(rr.boxes):
                    s["undecided"] = s.get("undecided", 0) + 1
                    continue
                b = rr.boxes[int(rr.boxes.conf.argmax())]
                p = [0.0, 0.0]
                p[int(b.cls)] = float(b.conf)  # a helmet box never counts as no_helmet evidence
            label = "no_helmet" if p[NH] >= a.no_helmet_thr else "helmet"
            s["two_wheelers"] += 1
            s[label] += 1
            col = (0, 200, 0) if label == "helmet" else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, f"{'NH' if label == 'no_helmet' else 'H'} {p[NH]:.2f}", (x1, max(14, y1 - 4)), 0, 0.6, col, 2)
            if crops_saved < 80:
                c2 = cv2.resize(crop, (160, max(1, int(160 * crop.shape[0] / max(1, crop.shape[1])))))
                cv2.putText(c2, f"{label} {p[NH]:.2f}", (3, 13), 0, 0.42, col, 1)
                cv2.imwrite(str(od / "crops" / f"{fi:06d}_{x1}_{label}.jpg"), c2)
                crops_saved += 1
        if frames_saved < 6 and len(boxes) >= 3:
            cv2.imwrite(str(od / "frames" / f"{fi:06d}.jpg"), vis)
            frames_saved += 1
    s["no_helmet_share"] = round(s["no_helmet"] / max(1, s["two_wheelers"]), 3)
    (od / "summary.json").write_text(json.dumps(s, indent=1))
    overall[tag] = s
    print(json.dumps(s), flush=True)
(OUT / "summary_all.json").write_text(json.dumps(overall, indent=1))
