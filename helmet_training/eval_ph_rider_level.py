#!/usr/bin/env python3
"""
Out-of-domain, RIDER-level test on the Motorcycle Helmet Detection Dataset PH (Philippine elevated CCTV).
PH was NOT used for training (its boxes cover the whole rider+motorcycle, not the head), so every PH frame is unseen.

For each PH rider box:  rider_full_face / rider_half_face -> helmet,  rider_no_helmet -> no_helmet,
rider_helmet_invalid -> skipped (ambiguous).  The model's head detections whose centre lies inside the
rider box's upper 45 % are collected; the highest-confidence one decides the rider's predicted status.

    .venv/bin/python helmet_training/eval_ph_rider_level.py --weights <w> --imgsz 960 --conf 0.35 --tag <tag>
"""
import argparse, json, random
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
PH = ROOT / "Motorcycle Helmet Detection Dataset PH.v2i.yolov8"
MAP = {0: 0, 1: 0, 3: 1}  # full_face, half_face -> helmet ; no_helmet -> no_helmet ; 2 (invalid) skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", required=True, help="conf or conf_helmet,conf_no_helmet"); ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--anywhere", action="store_true", help="accept detections centred anywhere in the rider box (for rider-box-trained models)"); ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.weights)
    th = [float(x) for x in str(a.conf).split(",")]; th = th * 2 if len(th) == 1 else th
    # one Roboflow copy per source frame, only frames with at least one rider label
    seen, items = set(), []
    for split in ("test", "valid", "train"):
        for p in sorted((PH / split / "images").iterdir()):
            base = p.name.split(".rf.")[0]
            if base in seen: continue
            L = [l.split() for l in (PH / split / "labels" / (p.stem + ".txt")).read_text().splitlines() if l.strip()]
            if any(int(l[0]) in MAP for l in L):
                seen.add(base); items.append((p, L))
    random.Random(0).shuffle(items); items = items[: a.n]
    res = Counter(); per_video = {}
    for p, L in items:
        im = cv2.imread(str(p)); h, w = im.shape[:2]
        r = m.predict(im, imgsz=a.imgsz, conf=min(th), device=a.device, verbose=False)[0]
        dets = [(b.xyxy[0].tolist(), float(b.conf), int(b.cls)) for b in r.boxes if float(b.conf) >= th[int(b.cls)]]
        vid = p.name.split("_mp4")[0]
        for l in L:
            c = int(l[0])
            if c not in MAP: continue
            gt = MAP[c]; x, y, bw, bh = map(float, l[1:5])
            x1, y1, x2 = (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w
            y_top = y1 + (1.0 if a.anywhere else 0.45) * bh * h
            cand = [d for d in dets if x1 <= (d[0][0] + d[0][2]) / 2 <= x2 and y1 - 0.05 * bh * h <= (d[0][1] + d[0][3]) / 2 <= y_top]
            if not cand: out = "missed"
            else:
                pc = max(cand, key=lambda d: d[1])[2]
                out = "correct" if pc == gt else ("FALSE_NO_HELMET" if gt == 0 else "FALSE_HELMET")
            key = ("helmet" if gt == 0 else "no_helmet")
            res[(key, out)] += 1
            per_video.setdefault(vid, Counter())[(key, out)] += 1
    def pack(c):
        o = {}
        for k in ("helmet", "no_helmet"):
            n = sum(v for (kk, _), v in c.items() if kk == k)
            o[k] = {"n_riders": n, **{out: c[(k, out)] for out in ("correct", "missed", "FALSE_NO_HELMET", "FALSE_HELMET") if c[(k, out)]},
                    "rider_recall(correct/n)": round(c[(k, "correct")] / n, 3) if n else None}
        return o
    rep = {"weights": a.weights, "anywhere_matching": a.anywhere, "conf": a.conf, "imgsz": a.imgsz, "frames": len(items), "overall": pack(res),
           "per_video": {k: pack(v) for k, v in per_video.items()},
           "note": "PH frames are 416x416 (stretched) elevated-CCTV; heads are typically 5-12 px. Out-of-domain stress test."}
    out = ROOT / "helmet_training/reports/eval" / a.tag; out.mkdir(parents=True, exist_ok=True)
    (out / ("ph_rider_level_anywhere.json" if a.anywhere else "ph_rider_level.json")).write_text(json.dumps(rep, indent=1)); print(json.dumps(rep["overall"], indent=1))


if __name__ == "__main__":
    main()
