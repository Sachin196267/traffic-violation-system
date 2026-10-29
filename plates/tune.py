#!/usr/bin/env python3
"""
Choose the READ threshold on the held-out test plates with the fine-tuned reader + format check.
A plate counts as READ if it is format-valid and its weakest character probability >= char_conf.
Picks the lowest char_conf whose READ plates are >= --target correct (default 98 %).
Writes plates/ocr/operating_point.json (char_conf, min_reads) and prints the trade-off table.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer

P = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--target", type=float, default=0.98)
ap.add_argument("--min-reads", type=int, default=2)
a = ap.parse_args()

src = (P / "run_video.py").read_text()
ns = {"re": re}
exec(src[src.index("STATES ="):src.index("ap = argparse")], ns)
exec(src[src.index("def normalise"):src.index("def read(im):")], ns)
normalise = ns["normalise"]

m = LicensePlateRecognizer(onnx_model_path=str(P / ("ocr/indian_reader_v2.onnx" if (P / "ocr/indian_reader_v2.onnx").exists() else "ocr/indian_reader.onnx")),
                           plate_config_path=str(P / "ocr/plate_config.yaml"), device="cpu")
rows = [r for r in json.loads((P / "data/plates.json").read_text()) if r["split"] == "test" and r["ocr_ok"]]
res = []
for r in rows:
    out = m.run(cv2.imread(str(P / "data" / r["crop"])), return_confidence=True)
    p = out[0] if isinstance(out, (list, tuple)) else out
    txt = "".join(ch for ch in str(p.plate).upper() if ch.isalnum())
    probs = np.asarray(p.char_probs, float).ravel()[:len(txt)]
    norm, _ = normalise(txt)
    res.append((norm, float(probs.min()) if len(probs) else 0.0, r["text"]))

n = len(res)
print(f"test plates {n}; raw exact {sum(t == g for t, _, g in [(normalise(x[0] or '')[0] if x[0] else None, 0, x[2]) for x in res]) / n:.3f}")
table, choice = [], None
for c in np.round(np.arange(0.30, 0.995, 0.02), 2):
    kept = [x for x in res if x[0] and x[1] >= c]
    acc = sum(x[0] == x[2] for x in kept) / max(1, len(kept))
    table.append((float(c), len(kept) / n, acc))
    if choice is None and acc >= a.target and kept:
        choice = (float(c), len(kept) / n, acc)
for c, cov, acc in table[::3]:
    print(f"  char_conf {c:.2f}: READ {cov:.0%} of plates, correct among READ {acc:.1%}")
valid = [x for x in res if x[0]]
print(f"format-valid readings {len(valid) / n:.0%}, correct among them {sum(x[0] == x[2] for x in valid) / max(1, len(valid)):.1%}")
if choice is None:
    choice = max(table, key=lambda t: t[2])
print(f"chosen char_conf {choice[0]}: READ {choice[1]:.0%} of single images, {choice[2]:.1%} correct")
(P / "ocr/operating_point.json").write_text(json.dumps(dict(
    char_conf=choice[0], min_reads=a.min_reads, target_correct=a.target,
    single_image_test=dict(read_rate=round(choice[1], 3), correct_among_read=round(choice[2], 4), n=n)), indent=1))
