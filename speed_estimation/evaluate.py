#!/usr/bin/env python3
"""
Compare speeds.json with a reference (reference.json: {track id: {"speed_kmh": v}}).

  python speed_estimation/evaluate.py <run dir> [<run dir> ...] [--tag _cal] [--side left|right]
Reports median error, error of the median estimate, and how often the 90 % range contains the reference
(should be about 90 % if the ranges are honest), per confidence level.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--tag", default="", help="evaluate speeds<tag>.json")
ap.add_argument("--side", help="only vehicles on this carriageway (left/right)")
a_ = ap.parse_args()
for rd in a_.runs:
    rd = Path(rd)
    est = json.loads((rd / f"speeds{a_.tag}.json").read_text())["vehicles"]
    ref = json.loads((rd / "reference.json").read_text())
    rows = defaultdict(list)
    for tid, r in ref.items():
        if a_.side and r.get("side") != a_.side:
            continue
        e = est.get(tid)
        if not e or "median_kmh" not in e:
            continue
        v = r["speed_kmh"]
        lo, hi = e["range90_kmh"]
        rows[e["confidence"]].append((v, e["median_kmh"], lo <= v <= hi, (hi - lo) / 2))
        rows["ALL"].append(rows[e["confidence"]][-1])
    print(rd.name, a_.tag or "", f"(side {a_.side})" if a_.side else "")
    for k in ("ALL", "high", "medium", "low", "unknown", "stopped"):
        if not rows[k]:
            continue
        a = np.array([(v, m, c, hw) for v, m, c, hw in rows[k]], float)
        err = a[:, 1] - a[:, 0]
        print(f"  {k:8s} n={len(a):3d} | median error {np.median(err):+6.1f} km/h | mean |error| {np.mean(np.abs(err)):5.1f} km/h "
              f"({np.mean(np.abs(err) / a[:, 0]) * 100:4.1f} %) | 90% range contains truth {a[:, 2].mean() * 100:5.1f} % | "
              f"typical range +/- {np.median(a[:, 3]):4.1f} km/h")
