#!/usr/bin/env python3
"""
Stage 2: camera calibration from the image itself (GeoCalib: focal length + gravity direction, with uncertainty).

  python speed_estimation/calibrate.py <video> [--frames 5]

Runs GeoCalib on several frames spread over the processed segment and combines them (median; the spread
between frames is added to the reported uncertainty). Writes <out>/<video>/calib.json.
"""
import argparse, json, math, warnings
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")
from geocalib import GeoCalib  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default=str(ROOT / "speed_estimation/runs"))
ap.add_argument("--frames", type=int, default=5)
args = ap.parse_args()

vp = Path(args.video)
out = Path(args.out) / vp.stem
tr = json.loads((out / "tracks.json").read_text())
cap = cv2.VideoCapture(str(vp))
fps = tr["fps_video"]
times = [t for trk in tr["tracks"].values() for t in trk["t"]] or [tr["t0"]]
t0, t1 = min(times), max(times)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = GeoCalib().to(dev)
res = []
for k in range(args.frames):
    t = t0 + (t1 - t0) * (k + 0.5) / args.frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, f = cap.read()
    if not ok:
        continue
    img = torch.tensor(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(dev)
    r = model.calibrate(img)
    res.append({"f": float(r["camera"].f[0, 0]), "cx": float(r["camera"].c[0, 0]), "cy": float(r["camera"].c[0, 1]),
                "roll": float(r["gravity"].rp[0, 0]), "pitch": float(r["gravity"].rp[0, 1]),
                "sd_f": float(r["focal_uncertainty"].flatten()[0]), "sd_roll": float(r["roll_uncertainty"].flatten()[0]),
                "sd_pitch": float(r["pitch_uncertainty"].flatten()[0])})
if not res:
    raise SystemExit("no frames could be read for calibration")
med = {k: float(np.median([r[k] for r in res])) for k in res[0]}
# total uncertainty = typical per-frame model uncertainty (+) disagreement between frames
spread = {k: float(np.std([r[k] for r in res])) for k in ("f", "roll", "pitch")}
calib = {"W": tr["W"], "H": tr["H"], "f": med["f"], "cx": med["cx"], "cy": med["cy"], "roll": med["roll"], "pitch": med["pitch"],
         "sd_f": math.hypot(med["sd_f"], spread["f"]), "sd_roll": math.hypot(med["sd_roll"], spread["roll"]),
         "sd_pitch": math.hypot(med["sd_pitch"], spread["pitch"]), "per_frame": res,
         "hfov_deg": math.degrees(2 * math.atan(tr["W"] / 2 / med["f"]))}
(out / "calib.json").write_text(json.dumps(calib, indent=1))
print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in calib.items() if k != "per_frame"}))
