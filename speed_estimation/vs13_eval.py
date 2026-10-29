#!/usr/bin/env python3
"""
Side-view test on the VS13 dataset (single cars passing a roadside camera at cruise-control speeds; the speed is
in the file name, e.g. Peugeot208_68.mp4 = 68 km/h). Uncalibrated mode (GeoCalib lens only).

  python speed_estimation/vs13_eval.py datasets/speed/vs13 [--limit 12]
The main vehicle of each clip is its longest car track. Results -> speed_estimation/runs/vs13_results.json
Note: cruise-control speeds come from the car's speedometer, which typically reads 2-5 % high.
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--resume", action="store_true", help="skip clips that already have speeds.json")
a = ap.parse_args()
clips = sorted((p for p in Path(a.folder).rglob("*") if p.suffix.lower() == ".mp4"), key=lambda p: int(re.findall(r"_(\d+)\.mp4$", p.name, re.I)[0]))
if a.limit:
    idx = np.linspace(0, len(clips) - 1, a.limit).round().astype(int)
    clips = [clips[i] for i in sorted(set(idx))]
py = sys.executable
rows = []
for c in clips:
    truth = int(re.findall(r"_(\d+)\.mp4$", c.name, re.I)[0])
    done = (HERE / "runs" / c.stem / "speeds.json").exists() and a.resume
    for cmd in ([] if done else ([py, HERE / "track.py", c, "--detector", "coco", "--fps", "30"], [py, HERE / "calibrate.py", c, "--frames", "3"],
                [py, HERE / "speed.py", c, "--no-video", "--min-span", "0.25"])):
        r = subprocess.run([str(x) for x in cmd], capture_output=True, text=True)
        if r.returncode:
            print(c.name, "failed:", r.stderr[-300:].strip().splitlines()[-1] if r.stderr else "")
            break
    run = HERE / "runs" / c.stem
    if not (run / "speeds.json").exists():
        rows.append({"clip": c.name, "truth": truth, "status": "failed"}); continue
    sp = json.loads((run / "speeds.json").read_text())
    tracks = json.loads((run / "tracks.json").read_text())["tracks"]
    cand = [(len(tracks[t]["t"]), t) for t, v in sp["vehicles"].items() if "median_kmh" in v and tracks[t]["cls"] == "car"]
    if not cand:
        rows.append({"clip": c.name, "truth": truth, "status": "no measurable car"}); print(c.name, truth, "no measurable car"); continue
    v = sp["vehicles"][max(cand)[1]]
    lo, hi = v["range90_kmh"]
    rows.append({"clip": c.name, "truth": truth, "median": v["median_kmh"], "range": [lo, hi], "confidence": v["confidence"],
                 "inside": lo <= truth <= hi, "camera": sp["summary"]["camera"]})
    print(f"{c.name}: truth {truth} | estimate {v['median_kmh']} ({lo:.0f}-{hi:.0f}) {v['confidence']} | inside {lo <= truth <= hi}", flush=True)
(HERE / "runs" / "vs13_results.json").write_text(json.dumps(rows, indent=1))
ok = [r for r in rows if "median" in r]
if ok:
    err = np.array([(r["median"] - r["truth"]) / r["truth"] for r in ok])
    shown = [r for r in ok if r["confidence"] in ("high", "medium", "low")]
    print(f"\nmeasured {len(ok)}/{len(rows)} clips | median error {np.median(err) * 100:+.1f}% | mean |error| {np.mean(np.abs(err)) * 100:.1f}% | "
          f"90% range contains truth {np.mean([r['inside'] for r in ok]) * 100:.0f}% | shown with a number {len(shown)} "
          f"(inside {np.mean([r['inside'] for r in shown]) * 100 if shown else 0:.0f}%)")
