#!/usr/bin/env python3
"""
Pass 2 of the red-light checker: signal state over time, stop line, crossings, ambulance exemption, decisions.

    python red_light/decide.py <video> [--camera cfg.json] [--render]

Needs <runs>/<video>/obs.json + heads.npz from observe.py. Every decision carries a confidence and reasons.

1. Signal state. Each signal head's crop is classified on every sampled frame (red / yellow / green /
   not_light), then smoothed with a hidden-Markov model: a lamp keeps its colour for seconds and changes in
   the order green -> yellow -> red -> green, so one misread frame cannot flip the state.
   A head is usable only if its lit face is seen (most frames confidently red/yellow/green).
2. Controlled flow + stop line (per shot). From a camera config when given ({"stop_line": [[x,y],[x,y]],
   "direction": [dx,dy], "light": [x1,y1,x2,y2]} in video pixels), else automatically:
   the direction whose traffic moves on green and stands still on red is the one this signal controls; the
   stop line is where the front of that queue stands during red. Automatic lines get a lower confidence.
3. Crossing. A vehicle moving in the controlled direction crosses the stop line (box bottom-centre, which is
   conservative for vehicles moving away from the camera).
4. Decision per crossing:
   VIOLATION  red (smoothed prob >= 0.9) for >= GRACE_S before the crossing, clear crossing, not an ambulance.
   NOT SURE   red but borderline (just turned red, weak light reading, automatic stop line with low support,
              vehicle turned sharply after the line - may be a free-left turn, or a possible ambulance).
   EXEMPT     ambulance (emergency vehicle) - never challaned.
   OK         crossed on green / yellow.
Outputs <runs>/<video>/decisions.json (+ <video>_redlight.mp4 with --render).
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = ["red", "yellow", "green", "unknown"]
GRACE_S = 1.0             # seconds after the light turns red before a crossing counts
P_RED = 0.9
TURN_DEG = 50             # heading change after the line that suggests a (free-left) turn
CLEAR_S = 5.0             # estimated stop line: ignore crossings this soon after red (vehicles still clearing the junction)

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--runs", default=str(ROOT / "red_light/runs"))
ap.add_argument("--camera", help="camera config json (stop line / direction / light)")
ap.add_argument("--light-weights", default=str(ROOT / "runs/signal_state/weights/last.pt"))
ap.add_argument("--amb-weights", default=str(ROOT / "runs/ambulance_cls/weights/last.pt"))
ap.add_argument("--amb-op", default=str(ROOT / "runs/ambulance_cls/operating_point.json"))
ap.add_argument("--render", action="store_true")
args = ap.parse_args()

vp = Path(args.video)
run = Path(args.runs) / vp.stem
obs = json.loads((run / "obs.json").read_text())
fps, step = obs["fps"], obs["step"]
cam = json.loads(Path(args.camera).read_text()) if args.camera else {}
from ultralytics import YOLO  # noqa: E402


# ---------------- 1. signal state ----------------
def light_probs():
    import torch
    npz = np.load(run / "heads.npz")
    m = YOLO(args.light_weights)
    order = [m.names[i] for i in range(len(m.names))]
    cols = [order.index(c) for c in ("red", "yellow", "green", "not_light")]
    out, net = {}, None
    for key in npz.files:
        crops = npz[key]
        if crops.ndim == 4 and crops.shape[1:3] == (64, 64):
            # crops are already the classifier's input size, so the predictor's resize/crop are identities: run the
            # network on big batches (same probabilities as m.predict, ~100x faster than one result object per crop)
            if net is None:
                m.predict(list(crops[:1]), imgsz=64, verbose=False)        # sets up the predictor's model + device
                net, dev = m.predictor.model, m.predictor.device
            P = []
            for i in range(0, len(crops), 1024):
                x = torch.from_numpy(np.ascontiguousarray(crops[i:i + 1024, ..., ::-1])).permute(0, 3, 1, 2).float().div(255)
                with torch.no_grad():
                    y = net(x.to(dev))
                P.append((y[0] if isinstance(y, (list, tuple)) else y).float().cpu().numpy()[:, cols])
            out[key.replace("_", ":", 1)] = np.concatenate(P) if P else np.zeros((0, 4))
            continue
        P = []
        for i in range(0, len(crops), 256):
            for r in m.predict(list(crops[i:i + 256]), imgsz=64, verbose=False):
                P.append(r.probs.data.cpu().numpy()[cols])
        out[key.replace("_", ":", 1)] = np.array(P)
    return out


def smooth(P, dt):
    """forward-backward over red/yellow/green/unknown; returns posteriors (n,4) and the most likely path."""
    stay = math.exp(-dt / 20.0)                 # a phase lasts tens of seconds
    move = 1 - stay
    T = np.full((4, 4), 1e-4)
    T[0, 0] = T[1, 1] = T[2, 2] = stay
    T[2, 1] = move * 0.9; T[1, 0] = move * 0.9; T[0, 2] = move * 0.9     # green->yellow->red->green
    T[0, 1] = move * 0.05                                                # red+amber before green (some signals)
    T[:3, 3] = move * 0.1; T[3, :3] = 0.05; T[3, 3] = 0.85
    T /= T.sum(1, keepdims=True)
    E = np.clip(P, 1e-3, 1)
    n = len(E)
    a = np.zeros((n, 4)); b = np.ones((n, 4))
    a[0] = E[0] / E[0].sum()
    for i in range(1, n):
        a[i] = E[i] * (a[i - 1] @ T); a[i] /= a[i].sum()
    for i in range(n - 2, -1, -1):
        b[i] = T @ (E[i + 1] * b[i + 1]); b[i] /= b[i].sum()
    post = a * b
    post /= post.sum(1, keepdims=True)
    return post, post.argmax(1)


heads = obs["heads"]
probs = light_probs() if heads else {}
head_state = {}
for hid, h in heads.items():
    if hid not in probs or len(probs[hid]) < 5:
        continue
    P = probs[hid]
    post, path = smooth(P, step / fps)
    lit = float(np.mean(post[:, :3].max(1) > 0.6))
    head_state[hid] = {"f": np.array(h["f"]), "post": post, "path": path, "lit_frac": lit, "n_det": h["n_det"],
                       "box": h["box"], "shot": h["shot"],
                       "source": h.get("source", "detector"),
                       # detector heads: seen >= 3 times and mostly lit; colour heads were already confirmed by
                       # their on/off phases, a red-only head is legitimately dark for part of the time
                       "usable": (lit >= 0.5 and h["n_det"] >= 3) or (h.get("source", "").startswith("colour") and lit >= 0.3)}


def state_at(hid, f):
    hs = head_state[hid]
    i = int(np.clip(np.searchsorted(hs["f"], f), 0, len(hs["f"]) - 1))
    return hs["post"][i], hs["path"][i], i


def red_onset(hid, i):
    """time (s) the current red phase started, walking back along the smoothed path"""
    hs = head_state[hid]
    j = i
    while j > 0 and hs["path"][j - 1] == 0:
        j -= 1
    return hs["f"][j] / fps, j == 0      # (onset time, onset not observed - red since the head appeared)


# ---------------- 2. motion, controlled flow, stop line ----------------
tracks = obs["tracks"]
for tid, tr in tracks.items():
    b = np.array(tr["box"]); tr["_t"] = np.array(tr["t"]); tr["_f"] = np.array(tr["f"])
    tr["_p"] = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]           # ground contact (bottom centre)
    tr["_h"] = np.maximum(b[:, 3] - b[:, 1], 1)


def velocities(tr, win=0.5):
    """(frame, position, velocity px/s, speed in box-heights/s) for every sample with a window of data"""
    t, p, h = tr["_t"], tr["_p"], tr["_h"]
    out = []
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] - win / 2); k = np.searchsorted(t, t[i] + win / 2, "right") - 1
        if k - j < 2 or t[k] - t[j] < win * 0.6:
            continue
        v = (p[k] - p[j]) / (t[k] - t[j])
        out.append((tr["_f"][i], p[i], v, float(np.linalg.norm(v) / h[i])))
    return out


VEL = {tid: velocities(tr) for tid, tr in tracks.items()}


def dominant_directions(shot):
    ang = [math.atan2(v[1], v[0]) for tid, tr in tracks.items() if tr["shot"] == shot
           for _, _, v, s in VEL[tid] if s > 0.3]
    if len(ang) < 20:
        return []
    hist, edges = np.histogram(ang, bins=24, range=(-math.pi, math.pi))
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)
    peaks = [i for i in range(24) if hist[i] >= hist[i - 1] and hist[i] >= hist[(i + 1) % 24] and hist[i] > 0.1 * sum(hist) / 3]
    dirs = []
    for i in sorted(peaks, key=lambda i: -hist[i])[:4]:
        c = (edges[i] + edges[i + 1]) / 2
        sel = [a for a in ang if abs(math.remainder(a - c, 2 * math.pi)) < math.radians(30)]
        m = math.atan2(np.mean(np.sin(sel)), np.mean(np.cos(sel)))
        if all(abs(math.remainder(m - d, 2 * math.pi)) > math.radians(30) for d in dirs):
            dirs.append(m)
    return [np.array([math.cos(d), math.sin(d)]) for d in dirs]


_PER_FRAME = {}


def moving_per_frame(d, shot):
    """frame -> number of vehicles moving in direction d (same for every signal head of the shot)"""
    key = (float(d[0]), float(d[1]), shot)
    if key not in _PER_FRAME:
        per_frame = defaultdict(int)
        for tid, tr in tracks.items():
            if tr["shot"] != shot:
                continue
            for f, _, v, s in VEL[tid]:
                if s > 0.3 and np.dot(v / (np.linalg.norm(v) + 1e-9), d) > math.cos(math.radians(35)):
                    per_frame[f] += 1
        _PER_FRAME[key] = per_frame
    return _PER_FRAME[key]


def flow_contrast(hid, d, shot):
    """how much traffic in direction d moves on green vs red (1 = only on green)"""
    moving = defaultdict(lambda: [0, 0])     # state -> [moving samples, frames]
    hs = head_state[hid]
    fset = hs["f"]
    red_since = None
    per_frame = moving_per_frame(d, shot)
    for i, f in enumerate(fset):
        st = hs["path"][i]
        if st == 0:
            red_since = f if red_since is None else red_since
            if (f - red_since) / fps < 2.0:    # vehicles already in the junction still clear it
                continue
        else:
            red_since = None
        moving[st][0] += per_frame.get(f, 0); moving[st][1] += 1
    g = moving[2][0] / max(moving[2][1], 1); r = moving[0][0] / max(moving[0][1], 1)
    enough = moving[2][1] * step / fps >= 3 and moving[0][1] * step / fps >= 3
    return (g - r) / (g + r + 1e-9), g, r, enough


def auto_stop_line(hid, d, shot):
    """front of the queue during red: stationary vehicles in the controlled lane(s), most advanced along d"""
    hs = head_state[hid]
    red_f = set(int(f) for f, s in zip(hs["f"], hs["path"]) if s == 0)
    pts, n_veh = [], 0
    for tid, tr in tracks.items():
        if tr["shot"] != shot:
            continue
        vs = VEL[tid]
        moved_d = [np.dot(v / (np.linalg.norm(v) + 1e-9), d) for _, _, v, s in vs if s > 0.3]
        if not moved_d or np.median(moved_d) < math.cos(math.radians(35)):
            continue                                           # never moves in the controlled direction
        still = [p for f, p, v, s in vs if s < 0.08 and int(f) in red_f]
        if len(still) >= max(3, int(1.0 * fps / step)):
            pts.append(np.median(still, 0)); n_veh += 1
    if n_veh < 2:
        return None
    pts = np.array(pts)
    n = np.array([-d[1], d[0]])
    along, lat = pts @ d, pts @ n
    # per lateral bin keep the most advanced stopped vehicle (queue head), then a line through them
    bins = np.linspace(lat.min() - 1, lat.max() + 1, 5)
    heads_pts = [pts[(lat >= a) & (lat < b)][np.argmax(along[(lat >= a) & (lat < b)])]
                 for a, b in zip(bins[:-1], bins[1:]) if np.any((lat >= a) & (lat < b))]
    heads_pts = np.array(heads_pts)
    s0 = float(np.max(heads_pts @ d))
    if len(heads_pts) >= 2 and np.ptp(heads_pts @ n) > 40:
        A = np.c_[heads_pts @ n, np.ones(len(heads_pts))]
        k, c = np.linalg.lstsq(A, heads_pts @ d, rcond=None)[0]    # along = k*lat + c
        k = float(np.clip(k, -0.5, 0.5))
        c = s0 - k * float(heads_pts[np.argmax(heads_pts @ d)] @ n)
    else:
        k, c = 0.0, s0
    lat_span = (float(lat.min()) - 60, float(lat.max()) + 60)
    p1 = n * lat_span[0] + d * (k * lat_span[0] + c); p2 = n * lat_span[1] + d * (k * lat_span[1] + c)
    conf = min(1.0, 0.5 + 0.1 * n_veh)
    return {"points": [p1.tolist(), p2.tolist()], "support_vehicles": n_veh, "confidence": round(conf, 2)}


def moving_away(tr):
    """net motion away from the camera: the ground point moves up the image and the box shrinks"""
    h, p = tr["_h"], tr["_p"]
    k = max(1, len(h) // 5)
    h0, h1 = float(np.median(h[:k])), float(np.median(h[-k:]))
    return h1 <= 0.8 * h0 and float(np.median(p[:k, 1]) - np.median(p[-k:, 1])) >= 0.3 * h0


def smooth_track(tr, max_step=1.5):
    """no jump of more than max_step box-heights between consecutive samples (a jump = the tracker swapped vehicles)"""
    h = tr["_h"]
    step_ = np.linalg.norm(np.diff(tr["_p"], axis=0), axis=1) / np.maximum(h[:-1], 1)
    return not len(step_) or float(step_.max()) <= max_step


def facing_setup(si, usable):
    """Used when the controlled direction cannot be confirmed from traffic (light red the whole shot, e.g. a car
    waiting at the red, or traffic moving on both red and green). A signal whose lit face we see controls the
    traffic moving away from the camera towards it, so that is taken as the controlled flow; the stop line is
    taken across the road at the middle of the vehicles standing still during red (median, so one parked or far
    vehicle cannot move it). Crossings judged this way are never challaned automatically - human review only."""
    hid = max(usable, key=lambda h: float(np.sum(head_state[h]["path"] == 0)))
    hs = head_state[hid]
    if np.sum(hs["path"] == 0) * step / fps < 3:
        return None                                   # red never seen for long enough: nothing to judge
    red_f = set(int(f) for f, s in zip(hs["f"], hs["path"]) if s == 0)
    sh = [(tid, tr) for tid, tr in tracks.items() if tr["shot"] == si and len(tr["_t"]) >= 5]
    away = [tr for _, tr in sh if moving_away(tr) and smooth_track(tr)]
    if not away:
        return None
    disp = np.array([tr["_p"][-1] - tr["_p"][0] for tr in away])
    d = disp.mean(0); d /= np.linalg.norm(d) + 1e-9
    sl = auto_stop_line(hid, d, si)          # front of the queue of vehicles that came this way and waited at red
    if sl:
        sl["confidence"] = min(sl["confidence"], 0.6)
        sl["source"] = f"automatic estimate: front of the queue at red ({sl['support_vehicles']} stopped vehicles)"
        return hid, d, sl
    xs = np.concatenate([tr["_p"][:, 0] for tr in away])
    x_lo, x_hi = float(xs.min()) - 0.1 * obs["W"], float(xs.max()) + 0.1 * obs["W"]
    queue = []
    for tid, tr in sh:
        still = [p for f, p, v, s in VEL[tid] if s < 0.08 and int(f) in red_f]
        if len(still) >= max(3, int(1.0 * fps / step)):
            m = np.median(still, 0)
            if x_lo <= m[0] <= x_hi:
                queue.append(m)
    if len(queue) < 2:
        return None
    y = float(np.median([q[1] for q in queue]))
    return hid, d, {"points": [[x_lo, y], [x_hi, y]], "support_vehicles": len(queue), "confidence": 0.6,
                    "source": f"automatic estimate: middle of the {len(queue)} vehicles waiting at red"}


cam_all = cam


def cam_for(si):
    """camera config for a shot: {"shots": {"<i>": {...}}} per shot, or one config for the whole video"""
    if "shots" in cam_all:
        return cam_all["shots"].get(str(si), {})
    return cam_all


shot_setup = {}
for si, sh in enumerate(obs["shots"]):
    cam = cam_for(si)
    usable = [h for h, s in head_state.items() if s["shot"] == si and s["usable"]]
    info = {"shot": si, "t": [sh["start_t"], sh["end_t"]], "usable_heads": usable, "reasons": []}
    shot_setup[si] = info
    if not usable:
        info["reasons"].append("no signal head with a visible lit lamp in this shot - red-light check not possible")
        continue
    if cam.get("light"):
        lb = cam["light"]
        hid = min(usable, key=lambda h: np.hypot((head_state[h]["box"][0] + head_state[h]["box"][2]) / 2 - (lb[0] + lb[2]) / 2,
                                                 (head_state[h]["box"][1] + head_state[h]["box"][3]) / 2 - (lb[1] + lb[3]) / 2))
        info["reasons"].append("signal head chosen by the camera configuration")
    else:
        hid = None
    dirs = [np.array(cam["direction"], float) / np.linalg.norm(cam["direction"])] if cam.get("direction") else dominant_directions(si)
    best = None
    for h in ([hid] if hid else usable):
        for d in dirs:
            c, g, r, enough = flow_contrast(h, d, si)
            if best is None or c > best[0]:
                best = (c, h, d, g, r, enough)
    if best is None:
        info["reasons"].append("too little moving traffic to tell which direction the signal controls")
        continue
    c, hid, d, g, r, enough = best
    info.update(head=hid, direction=d.round(3).tolist(), flow_contrast=round(c, 2), flow_green=round(g, 2), flow_red=round(r, 2))
    if cam.get("direction"):
        info["reasons"].append("controlled direction from the camera configuration")
        dir_ok = True
    else:
        dir_ok = enough and c >= 0.5
        info["reasons"].append(
            f"controlled direction found from traffic: {g:.1f} vehicles/frame move this way on green vs {r:.1f} on red"
            if dir_ok else "could not confirm the controlled direction from traffic (needs a red and a green phase with traffic)")
    if not dir_ok and not cam.get("direction") and not cam.get("stop_line"):
        fs = facing_setup(si, usable)
        if fs:
            hid, d, info["stop_line"] = fs
            info.update(head=hid, direction=d.round(3).tolist(), facing_prior=True)
            info["reasons"].append("controlled traffic taken as vehicles moving away from the camera towards the signal "
                                   "(its lit face points at the camera); stop line from the vehicles waiting at red")
            dir_ok = True
    if cam.get("stop_line"):
        info["stop_line"] = {"points": cam["stop_line"], "confidence": 1.0, "source": "camera configuration"}
    elif dir_ok and not info.get("facing_prior"):
        sl = auto_stop_line(hid, d, si)
        if sl:
            sl["source"] = f"automatic: front of the red-light queue ({sl['support_vehicles']} stopped vehicles)"
            info["stop_line"] = sl
        else:
            info["reasons"].append("no queue seen during red - stop line unknown")
    # a working signal changes colour: require >= 2 s of red AND >= 2 s of green/yellow in this shot, so a red sign,
    # a red vehicle or an overlay mistaken for a lamp can never produce a challan (at most NOT SURE)
    hs_ = head_state[hid]; dt_ = step / fps
    red_s = float(np.sum(hs_["path"] == 0) * dt_); go_s = float(np.sum((hs_["path"] == 1) | (hs_["path"] == 2)) * dt_)
    if hs_["source"] == "colour: red lamp switching on and off in phases":
        go_s = float(np.sum(hs_["path"] == 3) * dt_)   # only the red lamp is in view: 'dark' is the other phase
    info["signal_changes_seen"] = bool(red_s >= 2 and go_s >= 2)
    info["red_s"], info["green_yellow_s"] = round(red_s, 1), round(go_s, 1)
    if not info["signal_changes_seen"]:
        info["reasons"].append(f"signal not seen changing colour in this shot (red {red_s:.0f} s, green/yellow {go_s:.0f} s) "
                               "- crossings can only be NOT SURE")
    info["ready"] = bool(dir_ok and info.get("stop_line"))


# ---------------- 3. ambulance evidence ----------------
AMB_CROPS = defaultdict(list)
for _c in sorted((run / "amb_crops").glob("*.jpg")):
    AMB_CROPS[_c.stem.rsplit("_", 1)[0].replace("_", ":", 1)].append(str(_c))
amb, amb_op, _amb_scorer = {}, None, []


def ambulance_score(tid):
    """per track: emergency-vehicle score = mean of its 3 highest crop scores of P(ambulance or police or fire)
    from the CLIP probe (ambulance_training/train_probe.py), plus the most likely emergency type.
    Falls back to CLIP zero-shot ambulance probability if the probe is missing.
    Computed on first use: only vehicles that reach a red-light decision need it (scoring every vehicle's crops
    made the check take many minutes per video)."""
    global amb_op
    if tid in amb or not AMB_CROPS.get(tid):
        return
    if not _amb_scorer:
        import sys
        sys.path.insert(0, str(ROOT / "ambulance_training"))
        probe = ROOT / "runs/ambulance_cls/probe.npz"
        if probe.exists():
            from clip_features import EmergencyProbe
            ep = EmergencyProbe(probe)
            em = [ep.classes.index(c) for c in ("ambulance", "police", "fire")]
            _amb_scorer.append(lambda paths: [(float(p[em].sum()), ep.classes[em[int(np.argmax(p[em]))]])
                                              for p in ep.predict(paths)])
            default_op = {"t_exempt": 0.87, "t_review": 0.41, "model": "CLIP probe"}
        else:
            from clip_features import Clip
            clip = Clip()
            _amb_scorer.append(lambda paths: [(p["ambulance"], "ambulance") for p in clip.score(paths)[0]])
            default_op = {"t_exempt": 0.9, "t_review": 0.5, "model": "CLIP ViT-B/16 zero-shot"}
        amb_op = json.loads(Path(args.amb_op).read_text()) if Path(args.amb_op).exists() else default_op
    v = sorted(_amb_scorer[0](AMB_CROPS[tid]))[-3:]
    types = [ty for _, ty in v]
    amb[tid] = (float(np.mean([s for s, _ in v])), max(set(types), key=types.count))


def amb_status(tid):
    """('exempt' | 'review' | None, score, emergency type)"""
    ambulance_score(tid)
    if tid not in amb or amb_op is None:
        return None, None, None
    p, ty = amb[tid]
    if p >= amb_op["t_exempt"]:
        return "exempt", p, ty
    if p >= amb_op["t_review"]:
        return "review", p, ty
    return None, p, ty


# ---------------- 4. crossings and decisions ----------------
decisions = []
for tid, tr in tracks.items():
    info = shot_setup.get(tr["shot"])
    if not info or not info.get("ready"):
        continue
    d = np.array(info["direction"]); hid = info["head"]
    (x1, y1), (x2, y2) = info["stop_line"]["points"]
    nrm = np.array([-(y2 - y1), x2 - x1], float); nrm /= np.linalg.norm(nrm)
    if np.dot(nrm, d) < 0:
        nrm = -nrm
    sd = (tr["_p"] - np.array([x1, y1])) @ nrm                    # signed distance past the line (px)
    lat = (tr["_p"] - np.array([x1, y1])) @ np.array([x2 - x1, y2 - y1]) / np.hypot(x2 - x1, y2 - y1)
    seg_len = np.hypot(x2 - x1, y2 - y1)
    cross = [i for i in range(1, len(sd)) if sd[i - 1] < 0 <= sd[i]]
    if not cross:
        continue
    i = cross[0]
    if not (-0.25 * seg_len <= lat[i] <= 1.25 * seg_len):
        continue                                                   # crossed the line's extension, not the lane
    h = float(np.median(tr["_h"]))
    before = int(np.sum(sd[:i] < 0)); after_travel = float(sd.max())
    vs = [v for f, p, v, s in VEL[tid] if s > 0.3]
    if info.get("facing_prior"):     # perspective: controlled traffic converges away from the camera, no single direction
        heading_ok = moving_away(tr) and smooth_track(tr)
    else:
        heading_ok = bool(vs) and np.dot(np.mean([v / np.linalg.norm(v) for v in vs], 0), d) > math.cos(math.radians(45))
    if not heading_ok:
        continue                                                   # cross traffic, not the controlled flow
    # the crossing happened between the last sample before the line and the first after it (a gap when the
    # vehicle is hidden, e.g. behind a pole): be conservative - red must hold at BOTH ends, and "red for" is
    # measured from the earlier one
    f_cross = int(tr["_f"][i]); t_cross = f_cross / fps
    f_before = int(tr["_f"][i - 1]); gap = (f_cross - f_before) / fps
    post, st, k = state_at(hid, f_cross)
    post_b, st_b, k_b = state_at(hid, f_before)
    if st == 0 and st_b != 0:
        post, st, k = post_b, st_b, k_b          # not red yet when last seen before the line
    elif st == 0:
        post = np.minimum(post, post_b) if post_b[0] < post[0] else post
    onset, onset_unseen = red_onset(hid, k_b if st == 0 else k) if st == 0 else (None, False)
    t_ref = f_before / fps
    if st == 0 and info.get("facing_prior") and not onset_unseen and t_ref - onset < CLEAR_S:
        continue          # the estimated line may lie across the junction: vehicles that entered on green still clear it
    cls = max(set(tr["cls"]), key=tr["cls"].count)
    amb_s, amb_p, amb_t = amb_status(tid)
    after = [v for f, p, v, s in VEL[tid] if f > f_cross and s > 0.3]
    turn = None
    if after and not info.get("facing_prior"):     # vehicles past the line are far and small here: heading too noisy
        va = np.mean([v / np.linalg.norm(v) for v in after], 0)
        turn = math.degrees(math.acos(np.clip(np.dot(va / np.linalg.norm(va), d), -1, 1)))
    rec = {"track": tid, "class": cls, "t_cross": round(t_cross, 2), "frame": f_cross,
           "light": S[st], "p_red": round(float(post[0]), 3), "p_green": round(float(post[2]), 3),
           "red_for_s": None if onset is None else round(t_ref - onset, 2), "crossing_gap_s": round(gap, 2),
           "stop_line_source": info["stop_line"]["source"], "emergency_score": None if amb_p is None else round(amb_p, 3), "emergency_type": amb_t,
           "box": tr["box"][i], "reasons": []}
    R = rec["reasons"]
    track_ok = before >= 2 and after_travel >= 0.5 * h
    if st != 0:
        rec["decision"] = "OK"
        R.append(f"crossed the stop line on {S[st]} (smoothed signal probability {post[st]:.2f})")
        conf = float(post[st])
    else:
        R.append(f"signal red with probability {post[0]:.2f} when the vehicle crossed the stop line")
        R.append("red since before this shot started" if onset_unseen else
                 f"light had been red for at least {t_ref - onset:.1f} s (vehicle last seen before the line at {t_ref:.1f} s)")
        doubts = []
        if post[0] < P_RED:
            doubts.append("signal reading not certain enough")
        if not onset_unseen and t_ref - onset < GRACE_S:
            doubts.append(f"crossed within {GRACE_S:.0f} s of the light turning red (may have entered on yellow)")
        if not track_ok:
            doubts.append("crossing not clearly seen (vehicle tracked too briefly on one side of the line)")
        if not info["signal_changes_seen"]:
            doubts.append("the signal was not seen changing colour, so it is not confirmed to be a working signal")
        if info.get("facing_prior"):
            doubts.append("direction and stop line estimated automatically, not confirmed from a full signal cycle - needs a human check")
        elif info["stop_line"]["confidence"] < 0.8:
            doubts.append("stop line found automatically from few stopped vehicles")
        if turn is not None and turn > TURN_DEG:
            doubts.append(f"turned {turn:.0f} degrees after the line - may be a permitted (free-left) turn")
        if amb_s == "review":
            doubts.append(f"may be an emergency vehicle ({amb_t}, score {amb_p:.2f}) - needs a human check")
        conf = float(post[0]) * info["stop_line"]["confidence"] * (1.0 if track_ok else 0.6)
        if amb_s == "exempt":
            rec["decision"] = "EXEMPT"
            R.append(f"emergency vehicle: {amb_t} (score {amb_p:.2f}) - emergency vehicles are not challaned")
        elif doubts:
            rec["decision"] = "NOT SURE"
            R.extend(doubts)
        else:
            rec["decision"] = "VIOLATION"
            R.append(f"{cls} crossed {after_travel / h:.1f} vehicle-heights past the line in the controlled direction")
    rec["confidence"] = round(conf, 3)
    decisions.append(rec)

summary = {"video": obs["video"], "analysed_s": obs["analysed_s"],
           "counts": {k: sum(1 for d in decisions if d["decision"] == k) for k in ("VIOLATION", "NOT SURE", "EXEMPT", "OK")},
           "emergency_vehicles": {t: ty for t, (p, ty) in amb.items() if amb_op and p >= amb_op["t_exempt"]},
           "possible_emergency": {t: ty for t, (p, ty) in amb.items() if amb_op and amb_op["t_review"] <= p < amb_op["t_exempt"]},
           "ambulance_model": amb_op.get("model") if amb_op else None,
           "shots": [{k: v for k, v in s.items()} for s in shot_setup.values()],
           "heads": {h: {"usable": s["usable"], "lit_frac": round(s["lit_frac"], 2), "n_det": s["n_det"], "box": s["box"]}
                     for h, s in head_state.items()}}
(run / "decisions.json").write_text(json.dumps({"summary": summary, "decisions": decisions}, indent=1))
print(json.dumps(summary["counts"]), "| shots ready:", sum(1 for s in shot_setup.values() if s.get("ready")), "/", len(shot_setup))

# ---------------- 5. annotated video ----------------
if args.render:
    COL = {"VIOLATION": (0, 0, 255), "NOT SURE": (0, 170, 255), "EXEMPT": (255, 160, 0), "OK": (0, 190, 0)}
    LCOL = {0: (0, 0, 255), 1: (0, 200, 255), 2: (0, 200, 0), 3: (150, 150, 150)}
    by_track = {d["track"]: d for d in decisions}
    boxes = defaultdict(list)
    for tid, tr in tracks.items():
        for f, b in zip(tr["f"], tr["box"]):
            boxes[f].append((tid, b))
    cap = cv2.VideoCapture(str(vp))
    W, H = obs["W"], obs["H"]
    wr = cv2.VideoWriter(str(run / f"{vp.stem}_redlight.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (W, H))
    shot_of = lambda f: next((i for i, s in enumerate(obs["shots"]) if s["start_f"] <= f <= s["end_f"]), None)
    f = 0
    last_f = max((s["end_f"] for s in obs["shots"]), default=0)
    while f <= last_f:
        ok = cap.grab()
        if not ok:
            break
        if f % step:
            f += 1; continue
        ok, fr = cap.retrieve()
        si = shot_of(f); info = shot_setup.get(si, {})
        if info.get("stop_line"):
            p1, p2 = info["stop_line"]["points"]
            cv2.line(fr, tuple(map(int, p1)), tuple(map(int, p2)), (255, 255, 255), 2)
        if info.get("head") and info["head"] in head_state:
            post, st, _ = state_at(info["head"], f)
            b = list(map(int, head_state[info["head"]]["box"]))
            cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), LCOL[int(st)], 2)
            cv2.putText(fr, f"{S[int(st)]} {post[int(st)]:.2f}", (b[0], max(15, b[1] - 6)), 0, 0.5, LCOL[int(st)], 2)
        for tid, b in boxes.get(f, []):
            dcs = by_track.get(tid)
            if not dcs or f < dcs["frame"]:
                continue
            c = COL[dcs["decision"]]
            cv2.rectangle(fr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
            cv2.putText(fr, f"{dcs['decision']} {dcs['confidence']:.2f}", (int(b[0]), int(b[1]) - 6), 0, 0.55, c, 2)
        cv2.putText(fr, "white line = stop line | box colour on signal = smoothed state", (8, H - 10), 0, 0.5, (255, 255, 255), 1)
        wr.write(fr); f += 1
    wr.release()
    print("video:", run / f"{vp.stem}_redlight.mp4")
