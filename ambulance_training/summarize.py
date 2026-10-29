#!/usr/bin/env python3
"""Per-video ambulance summary from extract_crops.py output (CLIP zero-shot): tracks whose mean of the 3 best
crop scores is >= 0.9 (exempt) or 0.5-0.9 (review). Writes ambulance_training/summary.json and prints a table."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
out = {}
for d in sorted((ROOT / "crops").iterdir()):
    if not (d / "clip.json").exists():
        continue
    j = json.loads((d / "clip.json").read_text()); m = j["meta"]
    per = {}
    for f in j["files"]:
        per.setdefault(m[f]["track"], []).append(m[f]["p"]["ambulance"])
    sc = {t: float(np.mean(sorted(v)[-3:])) for t, v in per.items()}
    out[d.name] = {"vehicles": len(sc), "ambulance": sorted(t for t, s in sc.items() if s >= 0.9),
                   "review": sorted(t for t, s in sc.items() if 0.5 <= s < 0.9)}
    print(f"{d.name:14s} vehicles {len(sc):4d}  ambulance {len(out[d.name]['ambulance']):3d}  review {len(out[d.name]['review']):3d}")
(ROOT / "summary.json").write_text(json.dumps(out, indent=1))
