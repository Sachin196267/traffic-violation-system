#!/usr/bin/env python3
"""Label-free test of two-stage use: IDD motorcycle/rider box -> expanded crop -> helmet model.
Reports, per model, the fraction of two-wheeler crops that receive a helmet/no_helmet head box."""
import argparse, collections, json
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--stride", type=int, default=30)
ap.add_argument("--min-h", type=int, default=40, help="skip two-wheelers shorter than this (px)")
a = ap.parse_args()

idd = YOLO(str(ROOT / "runs/detect/runs/detect/idd_traffic_v2/weights/best.pt"))
models = {"FINAL": (ROOT / "runs/helmet_nohelmet_final/weights/best.pt", 0.45),
          "OLD": (ROOT / "runs/detect/runs/model2/helmet_nohelmet-2/weights/best.pt", 0.45)}
models = {k: (YOLO(str(p)), c) for k, (p, c) in models.items()}
out_dir = ROOT / "helmet_training/reports/crop_mode" / Path(a.video).stem
out_dir.mkdir(parents=True, exist_ok=True)

stats = {k: collections.Counter() for k in models}
cap = cv2.VideoCapture(a.video); i = 0; saved = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i % a.stride == 0:
        r = idd.predict(f, imgsz=960, conf=0.3, verbose=False)[0]
        H, W = f.shape[:2]
        for b in r.boxes:
            if r.names[int(b.cls)] not in ("motorcycle", "rider"): continue
            x1, y1, x2, y2 = map(int, b.xyxy[0]); h = y2 - y1
            if h < a.min_h: continue
            # extend upward: rider's head sits above a motorcycle box
            cx1, cy1, cx2, cy2 = max(0, x1 - h // 4), max(0, y1 - h), min(W, x2 + h // 4), y2
            crop = f[cy1:cy2, cx1:cx2]
            for k, (m, c) in models.items():
                p = m.predict(crop, imgsz=320, conf=c, verbose=False)[0]
                names = [p.names[int(x.cls)] for x in p.boxes]
                stats[k]["crops"] += 1
                stats[k]["any_head"] += bool(names)
                stats[k]["helmet"] += names.count("helmet"); stats[k]["no_helmet"] += names.count("no_helmet")
                if k == "FINAL" and saved < 24:
                    cv2.imwrite(str(out_dir / f"{i:05d}_{x1}.jpg"), p.plot()); saved += 1
    i += 1
res = {k: dict(v, head_found_rate=round(v["any_head"] / max(1, v["crops"]), 3)) for k, v in stats.items()}
(out_dir / "summary.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
