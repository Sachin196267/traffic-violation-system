#!/usr/bin/env python3
"""
Stage 3: vehicle speed as a range with a confidence level.

  python speed_estimation/speed.py <video> [--draws 300] [--no-video]

Needs tracks.json (track.py) and calib.json (calibrate.py) in speed_estimation/runs/<video>/.

1. Camera. Focal length and roll come from GeoCalib (with its uncertainty). The horizon - the parameter speed
   depends on most - comes from the road's vanishing point, where straight vehicle paths meet (falls back to
   GeoCalib's pitch when there are no consistent straight paths). Camera height: every tracked vehicle is
   modelled as a 3-D box of its class's typical size on the road, heading along its motion; the height is the
   one that makes the projected boxes as wide as the detected ones (median over vehicles).
2. Uncertainty. Camera draws = GeoCalib focal/roll uncertainty x bootstrap over vehicles (vanishing point and
   height re-estimated) + a common size-scale factor (fleet average size is uncertain) + Bayesian-bootstrap
   weights on each track's detections (tracking noise).
3. Speed. For each draw, box bottoms -> road-plane positions -> least-squares velocity per track.
   Reported: median and 90 % range (5th-95th percentile) in km/h, with a confidence level from the range width
   (high: within +-15 %, medium: +-25 %, low: +-40 %, otherwise 'unknown' - no number shown).
Outputs speeds.json and <video>_speed.mp4 (annotated) in the run folder.
"""
import argparse, json, math, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import background_to_first_frame, lane_points, map_points  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# typical size (width, length, height) m and per-vehicle spread (relative sd) of box width / height
CLASS = {
    "car": ((1.80, 4.40, 1.50), 0.07, 0.10),
    "motorcycle": ((0.80, 2.00, 1.55), 0.15, 0.10),   # with rider
    "autorickshaw": ((1.40, 2.60, 1.75), 0.06, 0.06),
    "bus": ((2.55, 11.0, 3.20), 0.05, 0.08),
    "truck": ((2.50, 9.00, 3.40), 0.08, 0.20),
}
SIZE_SCALE_SD = 0.05   # uncertainty of the fleet-average size (all classes together)
# model error measured in the cross-carriageway tests (calibrate on one side, check the other): the planar-road /
# camera model leaves a per-scene bias of about +-6 % and a per-vehicle scatter of about 4 %
MODEL_SCENE_SD, MODEL_VEHICLE_SD = 0.06, 0.04
LEVELS = [("high", 0.15), ("medium", 0.25), ("low", 0.40)]   # max half-width of the 90 % range, relative
SIGMA_M = 0.06         # per-detection box-size noise (log)

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default=str(ROOT / "speed_estimation/runs"))
ap.add_argument("--draws", type=int, default=300)
ap.add_argument("--boot", type=int, default=60, help="camera draws (bootstrap over vehicles + GeoCalib uncertainty)")
ap.add_argument("--min-span", type=float, default=0.8, help="min seconds a vehicle must be tracked")
ap.add_argument("--calib", help="one-time calibration file with a known along-road length (see calib_*.json)")
ap.add_argument("--focal-log-sd", type=float, default=0.5,
                help="uncalibrated: spread of the focal length around GeoCalib (log); 0.5 covers the 2x misses seen in tests")
ap.add_argument("--tag", default="", help="suffix for output files (speeds<tag>.json)")
ap.add_argument("--no-video", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

vp = Path(args.video)
run = Path(args.out) / vp.stem
tr = json.loads((run / "tracks.json").read_text())
cal = json.loads((run / "calib.json").read_text())
W, H = tr["W"], tr["H"]
cx, cy = cal["cx"], cal["cy"]

# ---------------- detections usable for geometry ----------------
tids, T, U, V, BW, BH, CLS = [], [], [], [], [], [], []
tracks = {}
for tid, t in tr["tracks"].items():
    rows = [(tt, b) for tt, b in zip(t["t"], t["box"]) if b[0] > 2 and b[2] < W - 3 and b[3] < H - 3 and b[1] > 2]
    if len(rows) < 6 or rows[-1][0] - rows[0][0] < args.min_span or t["cls"] not in CLASS:
        continue
    j = len(tracks)
    tracks[tid] = j
    for tt, (x1, y1, x2, y2) in rows:
        tids.append(j); T.append(tt); U.append((x1 + x2) / 2); V.append(y2); BW.append(x2 - x1); BH.append(y2 - y1)
        CLS.append(t["cls"])
tids, T, U, V, BW, BH = map(np.array, (tids, T, U, V, BW, BH))
NT = len(tracks)
if NT == 0:
    raise SystemExit("no usable vehicle tracks")
dims = np.array([CLASS[c][0] for c in CLS])                 # per detection (W, L, Hv)
sd_w = np.array([CLASS[c][1] for c in CLS]); sd_h = np.array([CLASS[c][2] for c in CLS])
cnt = np.bincount(tids, minlength=NT).astype(float)


def gdown(roll, pitch):
    """gravity (down) direction in camera coordinates (x right, y down, z forward); GeoCalib convention."""
    return np.array([math.sin(roll) * math.cos(pitch), math.cos(roll) * math.cos(pitch), -math.sin(pitch)])


def ground(u, v, f, roll, pitch, hc):
    """road-plane points (camera coords) of pixels; nan above the horizon."""
    g = gdown(roll, pitch)
    d = np.stack([(u - cx) / f, (v - cy) / f, np.ones_like(u)], -1)
    den = d @ g
    t = np.where(den > 1e-3, hc / np.where(den > 1e-3, den, 1), np.nan)
    return d * t[:, None], g


def track_velocity(P, w=None):
    """weighted least-squares velocity (3-vector, m/s) per track."""
    w = np.ones(len(T)) if w is None else w
    sw = np.bincount(tids, w, NT)
    tm = np.bincount(tids, w * T, NT) / sw
    dt = T - tm[tids]
    den = np.bincount(tids, w * dt * dt, NT)
    vel = np.stack([np.bincount(tids, w * dt * P[:, k], NT) for k in range(3)], -1) / den[:, None]
    return vel


def predicted_boxes(p):
    f, roll, pitch, hc = p
    P, g = ground(U, V, f, roll, pitch, hc)
    ok = np.isfinite(P[:, 0])
    P = np.where(ok[:, None], P, 0)
    vel = track_velocity(P)
    heading = vel - (vel @ g)[:, None] * g
    spd = np.linalg.norm(heading, axis=1)
    heading = heading / np.maximum(spd, 1e-6)[:, None]
    hd = heading[tids]
    lat = np.cross(g, hd)
    away = P - (P @ g)[:, None] * g                      # from the camera's foot point to the vehicle, on the road
    away /= np.maximum(np.linalg.norm(away, axis=1), 1e-6)[:, None]
    Wd, Ld, Hd = dims[:, 0], dims[:, 1], dims[:, 2]
    along = np.sign(np.sum(hd * away, 1)) * (np.abs(np.sum(hd * away, 1)) > 0.5)     # vehicle seen end-on
    side = np.sign(np.sum(lat * away, 1)) * (np.abs(np.sum(hd * away, 1)) <= 0.5)    # seen side-on
    # box centre: the visible bottom edge's midpoint is the detection's ground point
    C = P + hd * (along * Ld / 2)[:, None] + lat * (side * Wd / 2)[:, None]
    us, vs = [], []
    for a in (-0.5, 0.5):
        for b in (-0.5, 0.5):
            for c in (0.0, 1.0):
                X = C + hd * (a * Ld)[:, None] + lat * (b * Wd)[:, None] - g * (c * Hd)[:, None]
                z = np.maximum(X[:, 2], 1e-3)
                us.append(f * X[:, 0] / z + cx); vs.append(f * X[:, 1] / z + cy)
    us, vs = np.array(us), np.array(vs)
    pw, ph = us.max(0) - us.min(0), vs.max(0) - vs.min(0)
    valid = ok & (spd[tids] * 0 == 0) & (np.linalg.norm(vel, axis=1)[tids] > 0.5)   # moving vehicles only
    return pw, ph, valid


def vanishing_point(sel):
    """Road direction vanishing point from straight vehicle paths (box-bottom trajectories), RANSAC + refit."""
    lines, wts = [], []
    for j in sel:
        m = tids == j
        pts = np.c_[U[m], V[m]]
        if len(pts) < 6 or np.ptp(pts[:, 1]) + np.ptp(pts[:, 0]) < 40:
            continue
        c = pts.mean(0)
        _, sv, vt = np.linalg.svd(pts - c)
        if sv[1] > 0.15 * sv[0]:          # path not straight
            continue
        d = vt[0]
        n = np.array([-d[1], d[0]])
        lines.append((n[0], n[1], -n @ c, c, d)); wts.append(np.ptp(pts @ d))
    if len(lines) < 4:
        return None
    L = np.array([l[:3] for l in lines]); wts = np.array(wts)
    best = (0, None)
    for _ in range(300):
        i, k = rng.choice(len(L), 2, replace=False)
        x = np.cross(L[i], L[k])
        if abs(x[2]) < 1e-9:
            continue
        x = x[:2] / x[2]
        ang = [abs(np.dot((x - l[3]) / (np.linalg.norm(x - l[3]) + 1e-9), l[4])) for l in lines]
        inl = np.array(ang) > math.cos(math.radians(2.0))
        if inl.sum() > best[0]:
            best = (inl.sum(), inl)
    if best[1] is None or best[0] < max(4, 0.5 * len(L)):
        return None
    A = L[best[1]] * np.sqrt(wts[best[1]])[:, None]
    sol, *_ = np.linalg.lstsq(A[:, :2], -A[:, 2], rcond=None)   # point closest to all inlier lines
    return sol


def pitch_from_vp(vp, f, roll):
    a, b = (vp[0] - cx) / f, (vp[1] - cy) / f
    return math.atan(a * math.sin(roll) + b * math.cos(roll))


def camera_height(f, roll, pitch, sel, iters=5):
    """camera height that makes the modelled vehicle boxes as wide as the detected ones (median over vehicles)."""
    hc = 8.0
    for _ in range(iters):
        pw, ph, valid = predicted_boxes([f, roll, pitch, hc])
        r = np.where(valid, pw / np.maximum(BW, 1), np.nan)
        per_track = np.array([np.nanmedian(r[tids == j]) if np.any(valid[tids == j]) else np.nan for j in range(NT)])
        vals = per_track[sel]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return np.nan
        hc *= float(np.median(vals))
    return hc


calib = json.loads(Path(args.calib).read_text()) if args.calib else None


def calib_pairs(c):
    """(pixel a, pixel b, metres) pairs along the road from a calibration file, in tracks.json coordinates."""
    Hbg = background_to_first_frame(run) if c.get("coords") == "background" else np.eye(3)
    pairs = []
    if c["type"] == "distance":
        a, b = map_points(Hbg, c["points"])
        pairs.append((tuple(a), tuple(b), c["metres"]))
    elif c["type"] == "dashes":
        for pts in lane_points(c, Hbg).values():
            for i in range(len(pts)):
                for k in range(i + 1, len(pts)):
                    pairs.append((pts[i][1], pts[k][1], pts[k][0] - pts[i][0]))
    return pairs


def focal_from_calib(pairs, roll, vpt, sel, tol):
    """focal length at which the known along-road lengths come out right (camera height from vehicle widths)."""
    A = np.array([p[0] for p in pairs], float); B = np.array([p[1] for p in pairs], float)
    D = np.array([p[2] for p in pairs], float) * rng.normal(1.0, tol)
    def ratio(f):
        pitch = pitch_from_vp(vpt, f, roll) if vpt is not None else cal["pitch"]
        hc = camera_height(f, roll, pitch, sel)
        Pa, _ = ground(A[:, 0], A[:, 1], f, roll, pitch, hc)
        Pb, _ = ground(B[:, 0], B[:, 1], f, roll, pitch, hc)
        d = np.linalg.norm(Pa - Pb, axis=1)
        return float(np.nanmedian(np.log(d / D)))
    lo, hi = cal["f"] * 0.2, cal["f"] * 5.0
    rlo, rhi = ratio(lo), ratio(hi)
    if not (np.isfinite(rlo) and np.isfinite(rhi)) or rlo * rhi > 0:
        return np.nan
    for _ in range(30):
        mid = math.sqrt(lo * hi)
        rm = ratio(mid)
        if rm * rlo > 0:
            lo, rlo = mid, rm
        else:
            hi = mid
    return math.sqrt(lo * hi)


all_tracks = np.arange(NT)
vp0 = vanishing_point(all_tracks)
pairs0 = calib_pairs(calib) if calib else None
cams = []
for d in range(args.boot):
    sel = rng.integers(0, NT, NT) if d else all_tracks
    roll = float(rng.normal(cal["roll"], cal["sd_roll"])) if d else cal["roll"]
    vpt = vanishing_point(sel) if (d and vp0 is not None) else vp0
    if calib:
        pr = [pairs0[i] for i in rng.integers(0, len(pairs0), len(pairs0))] if (d and len(pairs0) > 3) else pairs0
        f = focal_from_calib(pr, roll, vpt, sel, calib.get("tolerance", 0.02) if d else 0.0)
        if not np.isfinite(f):
            continue
    else:
        f = float(cal["f"] * math.exp(rng.normal(0, max(args.focal_log_sd, cal["sd_f"] / cal["f"])))) if d else cal["f"]
    pitch = pitch_from_vp(vpt, f, roll) if vpt is not None else (float(rng.normal(cal["pitch"], cal["sd_pitch"])) if d else cal["pitch"])
    hc = camera_height(f, roll, pitch, sel)
    if d:  # real sizes vary per vehicle (~8 %); with few vehicles the bootstrap alone under-states this
        hc *= math.exp(rng.normal(0, 0.08 / math.sqrt(NT)))
    if np.isfinite(hc):
        cams.append([f, roll, pitch, hc])
cams = np.array(cams)
print("vanishing point", None if vp0 is None else [round(float(v), 1) for v in vp0], "camera draws", len(cams),
      "calibration", calib["type"] if calib else "none (GeoCalib focal)")

# ---------------- speeds ----------------
draws = []
for d in range(args.draws):
    f, roll, pitch, hc = cams[rng.integers(0, len(cams))]
    P, g = ground(U, V, f, roll, pitch, hc)
    ok = np.isfinite(P[:, 0])
    w = rng.exponential(1.0, len(T)) * ok                    # Bayesian bootstrap over detections
    vel = track_velocity(np.where(ok[:, None], P, 0), w)
    vel = vel - (vel @ g)[:, None] * g
    draws.append(np.linalg.norm(vel, axis=1) * 3.6 * rng.normal(1.0, SIZE_SCALE_SD)
                 * math.exp(rng.normal(0, MODEL_SCENE_SD)) * np.exp(rng.normal(0, MODEL_VEHICLE_SD, NT)))
draws = np.array(draws)                                       # (draws, tracks)

moving_cam = False  # track.py registers every frame to the first one; unregistrable footage is cut off there
res = {}
for tid, j in tracks.items():
    s = draws[:, j]
    s = s[np.isfinite(s)]
    if moving_cam or len(s) < args.draws // 2:
        res[tid] = {"status": "camera moving" if moving_cam else "no ground contact"}
        continue
    lo, med, hi = np.percentile(s, [5, 50, 95])
    rel = (hi - lo) / 2 / max(med, 1e-6)
    if hi < 5:
        level = "stopped"
    else:
        level = next((name for name, lim in LEVELS if rel <= lim), "unknown")
    ts = T[tids == j]
    res[tid] = {"cls": tr["tracks"][tid]["cls"], "median_kmh": round(float(med), 1), "range90_kmh": [round(float(lo), 1), round(float(hi), 1)],
                "confidence": level, "rel_halfwidth": round(float(rel), 3), "t_start": round(float(ts.min()), 2),
                "t_end": round(float(ts.max()), 2), "n": int(len(ts))}
f_, r_, p_, h_ = np.median(cams, 0)
summary = {"video": vp.name, "analysed_s": [tr["t0"], tr.get("t_end")], "stop_reason": tr.get("stop_reason"),
           "max_camera_shift_px": tr.get("max_shift_px"),
           "camera": {"f_px": round(float(f_), 1), "f_range": [round(float(v), 1) for v in np.percentile(cams[:, 0], [5, 95])],
                      "hfov_deg": round(math.degrees(2 * math.atan(W / 2 / f_)), 1), "roll_deg": round(math.degrees(r_), 2),
                      "pitch_deg": round(math.degrees(p_), 2), "height_m": round(float(h_), 2),
                      "height_range_m": [round(float(v), 2) for v in np.percentile(cams[:, 3], [5, 95])],
                      "geocalib_prior": {"f": round(cal["f"], 1), "sd_f": round(cal["sd_f"], 1), "pitch_deg": round(math.degrees(cal["pitch"]), 2)},
                      "vanishing_point": None if vp0 is None else [round(float(v), 1) for v in vp0],
                      "horizon_source": "vehicle paths" if vp0 is not None else "GeoCalib only (no consistent straight paths)",
                      "focal_source": f"calibration ({calib['type']})" if calib else "GeoCalib (uncalibrated, wide uncertainty)"},
           "vehicles": len(res), "by_confidence": {k: sum(1 for r in res.values() if r.get("confidence") == k)
                                                   for k in ("high", "medium", "low", "unknown", "stopped")}}
(run / f"speeds{args.tag}.json").write_text(json.dumps({"summary": summary, "vehicles": res}, indent=1))
print(json.dumps(summary))

# ---------------- annotated video ----------------
if not args.no_video:
    COL = {"high": (0, 200, 0), "medium": (0, 200, 255), "low": (0, 140, 255), "unknown": (160, 160, 160), "stopped": (200, 200, 200)}
    boxes = {}
    for tid, t in tr["tracks"].items():
        for tt, b in zip(t["t"], t.get("box_img", t["box"])):
            boxes.setdefault(round(tt, 4), []).append((tid, b))
    cap = cv2.VideoCapture(str(vp))
    fps = tr["fps_video"]
    step = tr["step"]
    times = sorted(boxes)
    wr = cv2.VideoWriter(str(run / f"{vp.stem}_speed{args.tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (W, H))
    fi = int(round(times[0] * fps)) if times else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    for tt in times:
        target = int(round(tt * fps))
        while fi < target:
            cap.grab(); fi += 1
        ok, fr = cap.read(); fi += 1
        if not ok:
            break
        placed = []   # label rectangles already drawn; a label that would cover one is left out (box still drawn)
        # speed numbers first, then nearest vehicles (lowest box bottom) first
        order = sorted(boxes[tt], key=lambda e: (res.get(e[0], {}).get("confidence") not in ("high", "medium", "low"), -e[1][3]))
        for tid, (x1, y1, x2, y2) in order:
            r = res.get(tid)
            if not r or "confidence" not in r:
                continue
            lv = r["confidence"]
            if y2 - y1 < 25:  # too small to label legibly
                cv2.rectangle(fr, (int(x1), int(y1)), (int(x2), int(y2)), COL[lv], 1)
                continue
            txt = {"stopped": "stopped", "unknown": "speed ?"}.get(lv, f"{r['range90_kmh'][0]:.0f}-{r['range90_kmh'][1]:.0f} km/h")
            (tw, th), _ = cv2.getTextSize(txt, 0, 0.6, 2)
            numeric = lv in ("high", "medium", "low")
            lab = (int(x1), int(y1) - th - (32 if numeric else 18), int(x1) + max(tw, 60) + 6, int(y1))
            free = not any(lab[0] < q[2] and q[0] < lab[2] and lab[1] < q[3] and q[1] < lab[3] for q in placed)
            cv2.rectangle(fr, (int(x1), int(y1)), (int(x2), int(y2)), COL[lv], 2 if free else 1)
            if not free:
                continue
            placed.append(lab)
            cv2.rectangle(fr, (lab[0], lab[1]), (lab[2], lab[3]), (0, 0, 0), -1)
            cv2.putText(fr, txt, (int(x1) + 3, int(y1) - 10), 0, 0.6, COL[lv], 2)
            if numeric:
                cv2.putText(fr, lv, (int(x1) + 3, int(y1) - th - 20), 0, 0.45, COL[lv], 1)
        legend = "speed: 90% range, colour = confidence (green high / yellow medium / orange low / grey unknown)"
        (lw, lh), _ = cv2.getTextSize(legend, 0, 0.55, 2)
        cv2.rectangle(fr, (4, H - lh - 22), (lw + 16, H - 4), (0, 0, 0), -1)
        cv2.putText(fr, legend, (10, H - 12), 0, 0.55, (255, 255, 255), 2)
        wr.write(fr)
    wr.release()
