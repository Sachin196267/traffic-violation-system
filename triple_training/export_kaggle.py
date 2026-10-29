#!/usr/bin/env python3
"""
Package the labelled crops (+ their degraded copies) and samples.json for the Kaggle training job
as a PRIVATE Kaggle dataset.  Run after build_dataset.py.

    python triple_training/export_kaggle.py --out <dir>
    kaggle datasets create -p <dir> --dir-mode zip        # first time (private by default)
    kaggle datasets version -p <dir> -m "more labels" --dir-mode zip   # later
"""
import argparse
import json
import shutil
from pathlib import Path

R = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--id", default="sachin196267/tvs-triple-crops")
a = ap.parse_args()

out = Path(a.out)
(out / "crops").mkdir(parents=True, exist_ok=True)
samples = json.loads((R / "samples.json").read_text())
rows = []
for s in samples:
    src = Path(s["path"])
    deg = R / "aug" / f"deg_{src.name}"
    for f in (src, deg):
        dst = out / "crops" / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
    rows.append({k: s[k] for k in ("key", "video", "label", "crop_h", "fold")} | {"file": src.name})
(out / "samples.json").write_text(json.dumps(rows))
(out / "dataset-metadata.json").write_text(json.dumps(
    {"title": "tvs-triple-crops", "id": a.id, "licenses": [{"name": "other"}]}, indent=1))
n = len(list((out / "crops").iterdir()))
print(f"{len(rows)} samples, {n} image files -> {out}")
