#!/usr/bin/env python3
"""
Compare helmet models on the held-out own-video test crops (datasets/helmet_nohelmet_ft_own/own_test)
and on the original clean test split.

Matching: a prediction matches a labelled head if IoU >= 0.3 (candidate boxes are approximate).
Reported per model: helmet / no_helmet precision & recall, false_no_helmet (helmeted head called no_helmet ->
wrong challan), false_helmet (bare head called helmet -> missed violation), missed heads.
Predictions that match no labelled head are counted separately ("unmatched") - they may be real heads the
candidate generator did not propose, so they are not treated as false positives.
"""
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "datasets/helmet_nohelmet_ft_own"
MODELS = {
    "APP_helmet_nohelmet-2": (ROOT / "runs/detect/runs/model2/helmet_nohelmet-2/weights/best.pt", 512, {0: 0.55, 1: 0.45}),
    "head_model_task1": (ROOT / "runs/helmet_nohelmet_final/weights/best.pt", 640, {0: 0.45, 1: 0.50}),
    "FINETUNED_own": (ROOT / "runs/helmet_nohelmet_ft_own/weights/last.pt", 640, {0: 0.45, 1: 0.50}),
}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def load_gt(img):
    lb = DS / "own_test/labels" / (img.stem + ".txt")
    import cv2
    H, W = cv2.imread(str(img)).shape[:2]
    out = []
    for line in lb.read_text().splitlines():
        k, cx, cy, w, h = map(float, line.split())
        out.append((int(k), [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]))
    return out


res = {}
imgs = sorted((DS / "own_test/images").glob("*.jpg"))
for tag, (w, sz, thr) in MODELS.items():
    m = YOLO(str(w))
    cm = np.zeros((3, 3), int)  # rows = predicted (helmet, no_helmet, none), cols = true (helmet, no_helmet, unlabelled)
    for img in imgs:
        gt = load_gt(img)
        r = m.predict(str(img), imgsz=sz, conf=0.2, device="mps", verbose=False)[0]
        preds = [(int(b.cls), list(map(float, b.xyxy[0]))) for b in r.boxes if float(b.conf) >= thr[int(b.cls)]]
        used = set()
        for gk, gb in gt:
            best, bj = 0.3, None
            for j, (pk, pb) in enumerate(preds):
                if j not in used and iou(gb, pb) >= best:
                    best, bj = iou(gb, pb), j
            if bj is None:
                cm[2, gk] += 1
            else:
                used.add(bj)
                cm[preds[bj][0], gk] += 1
        for j, (pk, _) in enumerate(preds):
            if j not in used:
                cm[pk, 2] += 1
    per = {}
    for k, n in ((0, "helmet"), (1, "no_helmet")):
        tp = cm[k, k]
        per[n] = {"P(labelled heads)": round(tp / max(1, cm[k, 0] + cm[k, 1]), 3), "R": round(tp / max(1, cm[:, k].sum()), 3),
                  "n_true": int(cm[:, k].sum())}
    res[tag] = {"per_class": per,
                "false_no_helmet(helmet->no_helmet)": f"{cm[1, 0]} / {cm[:, 0].sum()}",
                "false_helmet(no_helmet->helmet)": f"{cm[0, 1]} / {cm[:, 1].sum()}",
                "missed(h/nh)": f"{cm[2, 0]}/{cm[2, 1]}",
                "unmatched_preds(h/nh)": f"{cm[0, 2]}/{cm[1, 2]}",
                "confusion(pred rows h,nh,none x true cols h,nh,unlabelled)": cm.tolist()}
    v = m.val(data=str(DS / "data.yaml"), split="test", imgsz=sz, device="mps", verbose=False, plots=False,
              project=str(ROOT / "helmet_training/reports/own_eval/val"), name=tag, exist_ok=True)
    res[tag]["original_clean_test"] = {"mAP50": round(float(v.box.map50), 3), "mAP50-95": round(float(v.box.map), 3),
                                       "no_helmet_R": round(float(v.box.r[1]), 3) if len(v.box.r) > 1 else None}
    print(tag, json.dumps({k: v for k, v in res[tag].items() if not k.startswith("confusion")}), flush=True)

out = ROOT / "helmet_training/reports/own_eval"
out.mkdir(parents=True, exist_ok=True)
(out / "own_test_comparison.json").write_text(json.dumps(res, indent=1))
