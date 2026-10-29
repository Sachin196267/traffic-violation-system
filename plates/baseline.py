#!/usr/bin/env python3
"""
Baseline on the held-out test plates:
  detector  - existing models/license_plate_best.pt: recall at IoU >= 0.5, false boxes per image
  reader    - EasyOCR on the ground-truth plate crop: exact full-plate match, character accuracy
Writes plates/logs/baseline.json.
"""
import json
from pathlib import Path

import cv2
import numpy as np

R = Path(__file__).resolve().parent
rows = [r for r in json.loads((R / "data/plates.json").read_text()) if r["split"] == "test"]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


from ultralytics import YOLO  # noqa: E402
det = YOLO(str(R.parent / "models/license_plate_best.pt"))
hit = fp = 0
for i in range(0, len(rows), 16):
    chunk = rows[i:i + 16]
    for r, res in zip(chunk, det.predict([c["image"] for c in chunk], imgsz=640, conf=0.25, verbose=False)):
        boxes = res.boxes.xyxy.tolist()
        ok = [b for b in boxes if iou(b, r["box"]) >= 0.5]
        hit += bool(ok)
        fp += len(boxes) - (1 if ok else 0)
det_res = dict(recall=round(hit / len(rows), 3), false_boxes_per_image=round(fp / len(rows), 3), n=len(rows))
print("detector:", det_res)

import easyocr  # noqa: E402
reader = easyocr.Reader(["en"], gpu=False, verbose=False)
ocr_rows = [r for r in rows if r["ocr_ok"]]
exact = ced = tot = 0
for r in ocr_rows:
    im = cv2.imread(str(R / "data" / r["crop"]))
    txt = "".join(t for _, t, _ in reader.readtext(im, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", detail=1))
    exact += txt == r["text"]
    ced += edit(txt, r["text"])
    tot += len(r["text"])
ocr_res = dict(exact_plate=round(exact / len(ocr_rows), 3), char_accuracy=round(1 - ced / tot, 3), n=len(ocr_rows))
print("easyocr on true crops:", ocr_res)
(R / "logs/baseline.json").write_text(json.dumps(dict(detector=det_res, easyocr=ocr_res), indent=1))
