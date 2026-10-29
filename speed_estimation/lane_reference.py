#!/usr/bin/env python3
"""
Independent reference speeds from lane-marking dashes of known standard spacing (validation only; the
speed estimator itself never uses lane markings).

  python speed_estimation/lane_reference.py detect <video>         # background image + dash clusters to pick from
  python speed_estimation/lane_reference.py ref <video>            # reference speeds for run/<video>/tracks.json

Config: speed_estimation/references/<video stem>.json
  {"dash_m": 2.0, "cycle_m": 9.0, "standard": "...", "vp": [x, y], "split_m": 0.0,
   "lines": {"name": {"side": "left|right", "dashes": [[y_far, y_near|null], ...]}},  (far -> near, consecutive)
   "rows": [y_min, y_max]}
Each lane line gets s(y) = A + B / (y - yv) fitted to its dash ends (exact for a straight line on a flat road).
A vehicle's reference speed = Theil-Sen slope of s(box bottom) vs time, averaged over its carriageway's lines.
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import theilslopes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import background_to_first_frame, lane_points, map_points  # noqa: E402

ROOT = Path(__file__).resolve().parent
mode, video = sys.argv[1], Path(sys.argv[2])
run = ROOT / "runs" / video.stem

if mode == "detect":
    cap = cv2.VideoCapture(str(video)); n = int(cap.get(7))
    fr = []
    for k in range(60):
        cap.set(1, int(k * n / 60)); ok, f = cap.read()
        if ok:
            fr.append(f)
    bg = np.median(np.stack(fr), 0).astype(np.uint8)
    cv2.imwrite(str(run / "background.png"), bg)
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 165) & (hsv[..., 1] < 60)).astype(np.uint8) * 255
    sp = json.loads((run / "speeds.json").read_text())["summary"]["camera"] if (run / "speeds.json").exists() else None
    vp = sp["vanishing_point"] if sp and sp.get("vanishing_point") else [bg.shape[1] / 2, bg.shape[0] * 0.3]
    white[: int(vp[1]) + 20] = 0
    nlab, lab, st, cen = cv2.connectedComponentsWithStats(white, 8)
    rows = []
    for i in range(1, nlab):
        x, y, w, h, a = st[i]
        if a < 25 or h > bg.shape[0] * 0.3:
            continue
        ys, xs = np.where(lab == i)
        ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys])))
        if ev[1] < 4 * ev[0]:
            continue
        rows.append(((cen[i][0] - vp[0]) / (cen[i][1] - vp[1]), int(cen[i][0]), int(cen[i][1]), int(a), int(ys.min()), int(ys.max())))
    print("vanishing point", vp)
    for r in sorted(rows):
        print(f"slope {r[0]:+.3f}  centre ({r[1]},{r[2]})  area {r[3]}  y {r[4]}-{r[5]}")
    sys.exit()

cfg = json.loads((ROOT / "references" / f"{video.stem}.json").read_text())


def fit(pts):
    """pts: [(distance m, (x, y))] along one lane line -> s(y) = A + B / (y - yv)."""
    ys = np.array([p[1][1] for p in pts], float); ss = np.array([p[0] for p in pts], float)
    best = None
    for yv in np.arange(0.0, min(ys) - 3, 0.25):
        X = np.c_[np.ones_like(ys), 1.0 / (ys - yv)]
        coef, *_ = np.linalg.lstsq(X, ss, rcond=None)
        rms = float(np.sqrt(np.mean((X @ coef - ss) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, yv, coef)
    rms, yv, (A, B) = best
    return (lambda y: A + B / (np.asarray(y, float) - yv)), rms, yv


# dash coordinates were measured on the background image; tracks are in first-analysed-frame coordinates
Hbg = background_to_first_frame(run) if cfg.get("coords") == "background" else np.eye(3)
LP = lane_points(cfg, Hbg)
maps = {}
for name, ln in cfg["lines"].items():
    m, rms, yv = fit(LP[name])
    maps[name] = (m, ln["side"])
    print(f"{name}: fit rms {rms:.3f} m, horizon row {yv:.1f}")
y0, y1 = cfg["rows"]
for side in ("left", "right"):
    ms = [m for m, s in maps.values() if s == side]
    if len(ms) > 1:
        d = [float(m(y1) - m(y0)) for m in ms]
        print(f"{side}: rows {y0}->{y1} = " + ", ".join(f"{v:.2f} m" for v in d) + f" (spread {100 * (max(d) - min(d)) / np.mean(d):.1f}%)")

tr = json.loads((run / "tracks.json").read_text())
vx, vy = map_points(Hbg, [cfg["vp"]])[0]
print(f"background -> first frame: vanishing point moves by {np.hypot(vx - cfg['vp'][0], vy - cfg['vp'][1]):.1f} px")
ref = {}
for tid, t in tr["tracks"].items():
    pts = []
    for tt, (x1, yt, x2, yb) in zip(t["t"], t["box"]):
        if not y0 <= yb <= min(y1, tr["H"] - 4) or x1 < 3 or x2 > tr["W"] - 3:
            continue
        slope = ((x1 + x2) / 2 - vx) / (yb - vy)
        side = "right" if slope > cfg.get("split_m", 0) + 0.05 else ("left" if slope < cfg.get("split_m", 0) - 0.05 else None)
        ms = [m for m, s in maps.values() if s == side]
        if not ms:
            continue
        pts.append((tt, float(np.mean([m(yb) for m in ms])), side))
    if len(pts) < 8 or pts[-1][0] - pts[0][0] < 0.8 or len({p[2] for p in pts}) > 1:
        continue
    tt = np.array([p[0] for p in pts]); ss = np.array([p[1] for p in pts])
    slope, icpt, lo, hi = theilslopes(ss, tt, 0.95)
    ref[tid] = {"speed_kmh": round(abs(slope) * 3.6, 2), "n": len(pts), "span_s": round(float(tt[-1] - tt[0]), 2),
                "side": pts[0][2], "cls": t["cls"]}
(run / "reference.json").write_text(json.dumps(ref, indent=1))
sp = [r["speed_kmh"] for r in ref.values()]
print(f"reference speeds for {len(ref)} tracks; median {np.median(sp):.1f} km/h, range {min(sp):.0f}-{max(sp):.0f}" if sp else "no reference tracks")
