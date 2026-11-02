#!/usr/bin/env python3
"""
Evaluate a plate reader on the held-out test crops (exact plate, char accuracy, and accuracy
among plates the reader is confident about).
    python plates/eval_reader.py --hub cct-s-v2-global-model
    python plates/eval_reader.py --onnx runs/plate_ocr/.../model.onnx --config plates/indian_plate_config.yaml
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer

R = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--hub")
ap.add_argument("--onnx")
ap.add_argument("--config")
ap.add_argument("--split", default="test")
a = ap.parse_args()

m = (LicensePlateRecognizer(a.hub, device="cpu") if a.hub else
     LicensePlateRecognizer(onnx_model_path=a.onnx, plate_config_path=a.config, device="cpu"))
rows = [r for r in json.loads((R / "data/plates.json").read_text()) if r["split"] == a.split and r["ocr_ok"]]


def edit(x, y):
    d = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        prev, d[0] = d[0], i
        for j, cy in enumerate(y, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cx != cy))
    return d[-1]


def read(model, im):
    """-> (plate text A-Z0-9, weakest character probability)."""
    try:
        out = model.run(im, return_confidence=True)
    except Exception:
        out = model.run(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[..., None], return_confidence=True)
    p = out[0] if isinstance(out, (list, tuple)) else out
    txt = "".join(ch for ch in str(p.plate).upper() if ch.isalnum())
    probs = np.asarray(p.char_probs, float).ravel() if getattr(p, "char_probs", None) is not None else np.ones(1)
    return txt, float(probs[:max(1, len(txt))].min())


res = []
for r in rows:
    txt, c = read(m, cv2.imread(str(R / "data" / r["crop"])))
    res.append((txt == r["text"], edit(txt, r["text"]), len(r["text"]), c, txt, r["text"], r["source"]))
n = len(res)
exact = sum(x[0] for x in res) / n
chars = 1 - sum(x[1] for x in res) / sum(x[2] for x in res)
print(f"{a.hub or a.onnx}: exact {exact:.3f}  char {chars:.3f}  n={n}")
for s in ("State-wise_OLX", "google_images", "video_images"):
    sub = [x for x in res if x[6] == s]
    print(f"   {s:15s} exact {sum(x[0] for x in sub) / max(1, len(sub)):.3f}  n={len(sub)}")
for th in (0.5, 0.7, 0.8, 0.9):
    sub = [x for x in res if x[3] >= th]
    if sub:
        print(f"   min-char-conf>={th}: kept {len(sub) / n:.0%}, exact among kept {sum(x[0] for x in sub) / len(sub):.3f}")
print("   examples wrong:", [(x[4], x[5]) for x in res if not x[0]][:8])
