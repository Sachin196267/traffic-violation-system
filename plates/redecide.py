#!/usr/bin/env python3
"""Re-read saved plate crops (plates/dump/<video>/<tid>_<frame>.jpg) with the current reader and apply the current
decision rule + duplicate guard, without re-running detection.  python plates/redecide.py [--char-conf X]"""
import argparse, glob, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np
P = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--char-conf", type=float); ap.add_argument("--out"); a = ap.parse_args()
src = (P / "run_video.py").read_text()
from fast_plate_ocr import LicensePlateRecognizer
ns = {"re": re, "np": np, "Counter": Counter, "defaultdict": defaultdict, "json": json}
exec(src[src.index("STATES ="):src.index("ap = argparse")], ns)
op = json.loads((P / "ocr/operating_point.json").read_text())
if a.char_conf: op["char_conf"] = a.char_conf
ns["op"] = op
ns["reader"] = LicensePlateRecognizer(onnx_model_path=str(P / "ocr/indian_reader_v2.onnx"),
                                      plate_config_path=str(P / "ocr/plate_config.yaml"), device="cpu")
for fn in ("def normalise", "def read(im)", "def fmt", "def decide(rs)", "def dedupe"):
    i = src.index(fn); j = src.index("\ndef ", i + 5) if "\ndef " in src[i + 5:] else len(src)
    k = src.find("\n\n\n", i)
    exec(src[i:k], ns)
summary, reads_all = Counter(), []
for vd in sorted(glob.glob(str(P / "dump/*"))):
    by = defaultdict(list)
    for f in glob.glob(vd + "/*.jpg"):
        by[Path(f).stem.split("_")[0]].append(f)
    result = []
    for tid, fs in by.items():
        rs = []
        for f in fs:
            im = cv2.imread(f); raw, probs = ns["read"](im); norm, fixes = ns["normalise"](raw)
            rs.append((raw, norm, probs, im.shape[0] / 1.2, 0, fixes))
        d = ns["decide"](rs); d["vehicle"] = tid; d["video"] = Path(vd).name; result.append(d)
    result = ns["dedupe"](result)
    summary.update(r["decision"] for r in result)
    reads_all += [r for r in result if r["decision"] == "READ"]
print(f"char_conf {op['char_conf']}: vehicles with a plate crop {sum(summary.values())}: {dict(summary)}")
if a.out:
    Path(a.out).write_text(json.dumps(reads_all, indent=0))
