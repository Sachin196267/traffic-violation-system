#!/usr/bin/env python3
"""
Pass 1 of the red-light checker: watch the video once and cache everything the decision step needs.

    python red_light/observe.py <video> [--fps 10] [--max-sec 0] [--out red_light/runs]

- Shots: compilation / news clips cut between cameras. A cut (big change of the whole picture) starts a new
  shot; tracks, signal heads and the stop line are per shot.
- Vehicles: IDD detector + ByteTrack; per track the class, box and time of every sampled frame.
- Signal heads: traffic-light detections (IDD model every frame + COCO model every 5th frame, which finds far
  lights IDD misses) that stay in one place are grouped into heads. A colour pre-pass also finds heads the
  detectors miss (e.g. big close-up LED heads): a red lamp and a green lamp next to each other that are never
  lit together and each lit for part of the shot - a signal changing phase, not a tail light or a sign. From the first
  detection on, the head's box is cropped on EVERY sampled frame (a lamp changing colour is often missed by the
  detector for a few frames) and stored as a 64x64 square crop for the state classifier.
- Ambulance evidence: up to 10 crops per four-wheeler track (every ~0.5 s) for the ambulance classifier.

Writes <out>/<video>/obs.json, heads.npz (crops per head) and amb_crops/<track>_<k>.jpg.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VEHICLES = {"car", "bus", "truck", "autorickshaw", "motorcycle", "bicycle", "caravan", "trailer", "vehicle fallback"}
FOUR_WHEEL = {"car", "bus", "truck", "caravan", "trailer", "vehicle fallback"}

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default=str(ROOT / "red_light/runs"))
ap.add_argument("--fps", type=float, default=10.0, help="frames analysed per second")
ap.add_argument("--max-sec", type=float, default=0)
ap.add_argument("--imgsz", type=int, default=1280, help="detector size; lights are small, keep this high")
ap.add_argument("--weights", default=str(ROOT / "runs/idd_final/weights/best.pt"))
ap.add_argument("--prepass-only", action="store_true", help="only find shots + colour heads (debug image)")
ap.add_argument("--coco", default=str(ROOT / "yolo11s.pt"), help="second light detector ('' to skip)")
args = ap.parse_args()

vp = Path(args.video)
out = Path(args.out) / vp.stem
(out / "amb_crops").mkdir(parents=True, exist_ok=True)
det = YOLO(args.weights)
names = det.names
TL = next(k for k, v in names.items() if v == "traffic light")
keep_cls = [k for k, v in names.items() if v in VEHICLES] + [TL]
coco = YOLO(args.coco) if args.coco else None

cap = cv2.VideoCapture(str(vp))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, round(fps / args.fps))
IMGSZ = min(args.imgsz, int(np.ceil(max(W, H) / 32) * 32))     # never upscale small videos
last = n_frames if not args.max_sec else min(n_frames, int(args.max_sec * fps))


def square(crop, size=64):
    h, w = crop.shape[:2]
    s = max(h, w)
    c = np.zeros((s, s, 3), np.uint8)
    c[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)


def pad(b, f):
    x1, y1, x2, y2 = b
    pw, ph = (x2 - x1) * f, (y2 - y1) * f
    return max(0, int(x1 - pw)), max(0, int(y1 - ph)), min(W, int(x2 + pw)), min(H, int(y2 + ph))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def lamp_masks(fr):
    small = cv2.resize(fr, (SW, SH), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h < 8) | (h > 172)) & (sat > 110) & (v > 170)
    green = (h > 45) & (h < 100) & (sat > 90) & (v > 160)
    return red, green


def prepass():
    """shot cuts + colour-found signal heads (full-resolution boxes) per shot"""
    cap0 = cv2.VideoCapture(str(vp))
    cuts, prev, fi, sh = [], None, 0, -1
    acc = {}
    frames_small = []
    while fi < last:
        if not cap0.grab():
            break
        if fi % step:
            fi += 1; continue
        ok, fr = cap0.retrieve()
        if not ok:
            break
        th = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is None or float(np.mean(np.abs(th - prev))) > 35:
            cuts.append(fi); sh += 1
            acc[sh] = [np.zeros((SH, SW), np.int32), np.zeros((SH, SW), np.int32), 0]
        prev = th
        r, g = lamp_masks(fr)
        acc[sh][0] += r; acc[sh][1] += g; acc[sh][2] += 1
        frames_small.append((sh, np.packbits(r), np.packbits(g)))
        fi += 1
    found = defaultdict(list)
    sx, sy = W / SW, H / SH
    for sh, (cr, cg, n) in acc.items():
        if n < 20:
            continue
        comps = {}
        for name, c in (("r", cr), ("g", cg)):
            m = ((c >= 0.08 * n) & (c <= 0.97 * n)).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))       # LED dots -> one lamp
            k, lab, st, cen = cv2.connectedComponentsWithStats(m)
            comps[name] = [(lab == i, st[i], cen[i]) for i in range(1, k)
                           if 3 <= st[i][4] <= 0.01 * SW * SH                        # lamp-sized
                           and 0.4 <= st[i][2] / st[i][3] <= 2.5                      # round-ish
                           and st[i][4] >= 0.4 * st[i][2] * st[i][3]]                 # compact blob
        if not comps["r"] or not comps["g"]:
            continue
        for name in comps:   # keep the 30 largest candidates of each colour
            comps[name] = sorted(comps[name], key=lambda c: -c[1][4])[:30]
        idx = [i for i, fs in enumerate(frames_small) if fs[0] == sh]
        on = {"r": np.zeros((len(comps["r"]), len(idx)), bool), "g": np.zeros((len(comps["g"]), len(idx)), bool)}
        for j, i in enumerate(idx):
            for name, pk in (("r", 1), ("g", 2)):
                full = np.unpackbits(frames_small[i][pk])[:SW * SH].reshape(SH, SW).astype(bool)
                for ci, (mask, st, cen) in enumerate(comps[name]):
                    on[name][ci, j] = full[mask].mean() > 0.3
        dt = step / fps

        def phased(x):
            """lit for part of the shot, in on/off runs of seconds (a signal phase, not flicker or brake lights)"""
            if not 0.08 <= x.mean() <= 0.97:
                return False
            edges = np.flatnonzero(np.diff(x.astype(np.int8))) + 1
            runs = np.diff(np.r_[0, edges, len(x)]) * dt
            return len(runs) >= 2 and np.median(runs) >= 2.0

        used_r = set()
        for ri, (mr, sr, cr_) in enumerate(comps["r"]):
            for gi, (mg, sg, cg_) in enumerate(comps["g"]):
                size = max(sr[2], sr[3], sg[2], sg[3])
                if np.hypot(*(cr_ - cg_)) > 5 * size or max(sr[4], sg[4]) > 4 * min(sr[4], sg[4]):
                    continue
                a, b = on["r"][ri], on["g"][gi]
                if not (phased(a) and phased(b)) or (a & b).mean() > 0.03:
                    continue
                score = 2 + min(a.mean(), b.mean()) + (a ^ b).mean()     # red + green alternating: strongest
                x1 = min(sr[0], sg[0]) - size; y1 = min(sr[1], sg[1]) - size
                x2 = max(sr[0] + sr[2], sg[0] + sg[2]) + size; y2 = max(sr[1] + sr[3], sg[1] + sg[3]) + size
                found[sh].append((score, [max(0, x1 * sx), max(0, y1 * sy), min(W, x2 * sx), min(H, y2 * sy)], "red+green"))
                used_r.add(ri)
        for ri, (mr, sr, cr_) in enumerate(comps["r"]):   # red lamp alone (other lamps hidden / out of view)
            if ri in used_r or not phased(on["r"][ri]):
                continue
            size = max(sr[2], sr[3])
            x1, y1, x2, y2 = sr[0] - size, sr[1] - size, sr[0] + sr[2] + size, sr[1] + sr[3] + size
            found[sh].append((1 + on["r"][ri].mean(), [max(0, x1 * sx), max(0, y1 * sy), min(W, x2 * sx), min(H, y2 * sy)], "red"))
    for sh in found:       # overlapping candidates are the same head: keep the best alternating pair
        keep = []
        for sc, b, kind in sorted(found[sh], key=lambda x: -x[0]):
            if all(iou(b, k[0]) < 0.2 for k in keep):
                keep.append((b, kind))
        found[sh] = keep
    return cuts, found


SW = 480; SH = max(1, round(480 * H / W))
cut_frames, colour_heads = prepass()
cut_set = set(cut_frames)
if args.prepass_only:
    cap0 = cv2.VideoCapture(str(vp))
    for sh, cf in enumerate(cut_frames):
        cap0.set(cv2.CAP_PROP_POS_FRAMES, cf + step * 5); ok, fr = cap0.read()
        if ok:
            for b, kind in colour_heads.get(sh, []):
                cv2.rectangle(fr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 2)
            cv2.imwrite(str(out / f"prepass_shot{sh}.jpg"), fr)
    print({k: [([int(x) for x in b], kind) for b, kind in v] for k, v in colour_heads.items()})
    raise SystemExit
print(f"pre-pass: {len(cut_frames)} shots, colour-found signal heads: {dict((k, len(v)) for k, v in colour_heads.items())}", flush=True)

shots, tracks, heads, head_crops = [], {}, {}, {}
amb_count, amb_last = {}, {}
shot = -1
fi = 0
while fi < last:
    ok = cap.grab()
    if not ok:
        break
    if fi % step:
        fi += 1
        continue
    ok, fr = cap.retrieve()
    if not ok:
        break
    t = fi / fps
    cut = fi in cut_set or shot < 0
    if cut:
        shot += 1
        shots.append({"start_f": fi, "end_f": fi, "start_t": round(t, 3), "end_t": round(t, 3)})
        for b, kind in colour_heads.get(shot, []):
            hid = f"{shot}:L{len(heads)}"
            heads[hid] = {"shot": shot, "box": list(b), "n_det": 0, "f": [], "det": [], "first_f": fi, "conf": [],
                          "source": "colour: red and green lamps alternating" if kind == "red+green"
                          else "colour: red lamp switching on and off in phases"}
            head_crops[hid] = []
    shots[-1]["end_f"], shots[-1]["end_t"] = fi, round(t, 3)
    r = det.track(fr, imgsz=IMGSZ, conf=0.2, classes=keep_cls, persist=not cut, tracker="bytetrack.yaml",
                  verbose=False)[0]
    bx = r.boxes
    xyxy = bx.xyxy.tolist(); cl = bx.cls.int().tolist(); cf = bx.conf.tolist()
    ids = bx.id.int().tolist() if bx.id is not None else [None] * len(cl)
    lights = []
    if coco is not None and (fi // step) % 5 == 0:
        rc = coco.predict(fr, imgsz=IMGSZ, conf=0.15, classes=[9], verbose=False)[0]
        lights += [(b, p) for b, p in zip(rc.boxes.xyxy.tolist(), rc.boxes.conf.tolist())]
    for b, c, p, i in zip(xyxy, cl, cf, ids):
        if c == TL:
            lights.append((b, p))
            continue
        if i is None or p < 0.3:
            continue
        tid = f"{shot}:{i}"
        tr = tracks.setdefault(tid, {"shot": shot, "cls": [], "f": [], "t": [], "box": [], "conf": []})
        tr["cls"].append(names[c]); tr["f"].append(fi); tr["t"].append(round(t, 3))
        tr["box"].append([round(v, 1) for v in b]); tr["conf"].append(round(p, 3))
        if names[c] in FOUR_WHEEL and b[3] - b[1] >= 40 and amb_count.get(tid, 0) < 10 and t - amb_last.get(tid, -9) >= 0.5:
            x1, y1, x2, y2 = pad(b, 0.05)
            cv2.imwrite(str(out / "amb_crops" / f"{tid.replace(':', '_')}_{amb_count.get(tid, 0)}.jpg"), fr[y1:y2, x1:x2])
            amb_count[tid] = amb_count.get(tid, 0) + 1; amb_last[tid] = t
    # signal heads: static light positions within the shot
    for b, p in lights:
        best, bi = None, 0.2
        for hid, h in heads.items():
            if h["shot"] != shot:
                continue
            o = iou(b, h["box"])
            if o > bi:
                best, bi = hid, o
        if best is None:
            best = f"{shot}:L{len(heads)}"
            heads[best] = {"shot": shot, "box": list(b), "n_det": 0, "f": [], "det": [], "first_f": fi, "conf": [],
                           "source": "detector"}
            head_crops[best] = []
        h = heads[best]
        h["box"] = [0.8 * a + 0.2 * c for a, c in zip(h["box"], b)]
        h["n_det"] += 1; h["conf"].append(round(p, 3))
        h["_seen"] = fi
    for hid, h in heads.items():
        if h["shot"] != shot:
            continue
        x1, y1, x2, y2 = pad(h["box"], 0.25)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        h["f"].append(fi); h["det"].append(int(h.get("_seen") == fi))
        head_crops[hid].append(square(fr[y1:y2, x1:x2]))
    if (fi // step) % 100 == 0:
        print(f"{t:6.1f}s shot {shot} tracks {len(tracks)} heads {len(heads)}", flush=True)
    fi += 1

for h in heads.values():
    h.pop("_seen", None)
    h["box"] = [round(v, 1) for v in h["box"]]
obs = {"video": vp.name, "fps": fps, "step": step, "W": W, "H": H, "analysed_s": round(fi / fps, 2),
       "detector": Path(args.weights).name, "shots": shots, "tracks": tracks, "heads": heads}
(out / "obs.json").write_text(json.dumps(obs))
np.savez_compressed(out / "heads.npz", **{k.replace(":", "_"): np.array(v, np.uint8) for k, v in head_crops.items() if v})
print(f"done: {len(shots)} shots, {len(tracks)} tracks, {len(heads)} signal heads, {sum(amb_count.values())} ambulance crops")
