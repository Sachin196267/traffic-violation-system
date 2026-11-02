#!/usr/bin/env python3
"""
Compare IDD traffic detectors on the local IDD test subset (1 000 images, all in the Kaggle test split)
and render the same video frames with each model for visual review.

    .venv/bin/python helmet_training/compare_idd.py --new runs/idd_kaggle/best.pt
Outputs: helmet_training/reports/idd_compare/{metrics.json, metrics.md, frames/<model>_<video>.jpg}
"""
import argparse, glob, json
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--new", required=True)
ap.add_argument("--device", default="mps")
a = ap.parse_args()

MODELS = {
    "app_traffic.pt(1ep)": (ROOT / "models/traffic.pt", 640),
    "idd_traffic_v2": (ROOT / "runs/detect/runs/detect/idd_traffic_v2/weights/best.pt", 640),
    "NEW_yolo11s_960": (Path(a.new), 960),
}
DATA = ROOT / "datasets/kaggle/idd/IDDDetectionsYOLODataset/data.yaml"
OUT = ROOT / "helmet_training/reports/idd_compare"
(OUT / "frames").mkdir(parents=True, exist_ok=True)
KEY = ["motorcycle", "rider", "person", "car", "autorickshaw", "bus", "truck", "bicycle", "traffic light", "traffic sign", "animal"]

res = {}
for tag, (w, sz) in MODELS.items():
    m = YOLO(str(w))
    r = m.val(data=str(DATA), split="test", imgsz=sz, batch=8, device=a.device, plots=False, verbose=False,
              project=str(OUT / "val"), name=tag, exist_ok=True)
    names = r.names
    res[tag] = {"imgsz": sz, "P": round(float(r.box.mp), 3), "R": round(float(r.box.mr), 3),
                "mAP50": round(float(r.box.map50), 3), "mAP50-95": round(float(r.box.map), 3),
                "per_class": {names[c]: {"P": round(float(r.box.p[i]), 3), "R": round(float(r.box.r[i]), 3),
                                         "mAP50": round(float(r.box.ap50[i]), 3)}
                              for i, c in enumerate(r.box.ap_class_index)}}
    print(tag, {k: v for k, v in res[tag].items() if k != "per_class"}, flush=True)

    vids = {"mumbai": "videos/test/model2_test.mp4", "sdelhi": "videos/Traffic congestion*", "delhi4k": "videos/Original Traffic Sound*",
            "bombay": "videos/bombay traffic*"}
    for vk, pat in vids.items():
        c = cv2.VideoCapture(glob.glob(str(ROOT / pat))[0])
        c.set(cv2.CAP_PROP_POS_FRAMES, int(c.get(cv2.CAP_PROP_FRAME_COUNT) * 0.4))
        ok, f = c.read()
        if ok:
            im = m.predict(f, imgsz=1280, conf=0.3, device=a.device, verbose=False)[0].plot(line_width=2, font_size=10)
            h, w_ = im.shape[:2]
            cv2.imwrite(str(OUT / "frames" / f"{tag}_{vk}.jpg"), cv2.resize(im, (1280, int(h * 1280 / w_))))

(OUT / "metrics.json").write_text(json.dumps(res, indent=1))
tags = list(res)
md = ["| metric | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
for k in ("imgsz", "P", "R", "mAP50", "mAP50-95"):
    md.append(f"| {k} | " + " | ".join(str(res[t][k]) for t in tags) + " |")
md += ["", "Per-class recall / mAP50 (test)", "", "| class | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
for c in KEY:
    md.append(f"| {c} | " + " | ".join(
        f"{res[t]['per_class'].get(c, {}).get('R', '-')} / {res[t]['per_class'].get(c, {}).get('mAP50', '-')}" for t in tags) + " |")
(OUT / "metrics.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
