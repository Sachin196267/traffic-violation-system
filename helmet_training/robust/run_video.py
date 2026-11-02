#!/usr/bin/env python3
"""
Robust helmet check on any video: one decision per rider (helmet / no_helmet / not_sure).

  python helmet_training/robust/run_video.py <video> [--out out_dir] [--max-sec 120]

Pipeline (same geometry as training):
  IDD detector + ByteTrack -> motorcycle tracks confirmed by a 2nd detector (COCO), ridden or moving
     (+ overlapping rider boxes; bicycles, parked bikes skipped) -> head boxes (pose + head model, on an
  upscaled crop) -> head patch -> classifier (helmet / no_helmet / not_head) every sampled frame
  -> per person: size-weighted vote over all frames -> decision with the tuned thresholds
     (runs/helmet_robust/final/operating_point.json, picked on held-out videos so that
      "not sure" is about 1 rider in 30).
Writes <out>/<video>_helmet.mp4 (annotated) and <out>/<video>_riders.json.
"""
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
C = ("helmet", "no_helmet", "not_head")
MOTO_CONF = 0.45  # min detector confidence for a motorcycle to be checked
COLORS = {"helmet": (0, 200, 0), "no_helmet": (0, 0, 255), "not_sure": (0, 200, 255)}

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default=str(ROOT / "helmet_training/robust/video_out"))
ap.add_argument("--fps", type=float, default=5.0)
ap.add_argument("--max-sec", type=float, default=0)
ap.add_argument("--min-head", type=int, default=8)
ap.add_argument("--weights", default=str(ROOT / "runs/helmet_robust/final/weights/last.pt"))
args = ap.parse_args()

op = json.loads((Path(args.weights).parents[1] / "operating_point.json").read_text())
idd = YOLO(str(ROOT / "runs/idd_final/weights/best.pt"))
pose = YOLO(str(ROOT / "yolo11s-pose.pt"))
helm = YOLO(str(ROOT / "runs/helmet_nohelmet_v2_final/weights/best.pt"))
cls = YOLO(args.weights)
coco = YOLO(str(ROOT / "yolo11s.pt"))  # second-opinion motorcycle detector


def overlap(a, b):
    """Share of b's area covered by a."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def heads_batch(crops):
    ups = []
    for c in crops:
        h0, w0 = c.shape[:2]
        s = max(1.0, 384 / min(h0, w0))
        ups.append(((cv2.resize(c, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_CUBIC) if s > 1 else c), s))
    out = []
    for i in range(0, len(ups), 16):
        chunk = [u[0] for u in ups[i:i + 16]]
        for r, hr in zip(pose.predict(chunk, imgsz=640, conf=0.25, verbose=False), helm.predict(chunk, imgsz=640, conf=0.2, verbose=False)):
            heads = []
            if r.keypoints is not None:
                for pb, kp in zip(r.boxes.xyxy.tolist(), r.keypoints.data.tolist()):
                    vis = [k for k in kp[:5] if k[2] > 0.3]
                    sh = [kp[5], kp[6]] if kp[5][2] > 0.3 and kp[6][2] > 0.3 else None
                    sw = abs(sh[0][0] - sh[1][0]) if sh else 0
                    ears = abs(kp[3][0] - kp[4][0]) if kp[3][2] > 0.3 and kp[4][2] > 0.3 else 0
                    size = max(0.8 * sw, 1.4 * ears, 0.35 * (pb[2] - pb[0]))
                    cx = np.mean([k[0] for k in vis]) if vis else ((sh[0][0] + sh[1][0]) / 2 if sh else (pb[0] + pb[2]) / 2)
                    heads.append([cx - size / 2, pb[1], cx + size / 2, pb[1] + 1.1 * size])
            for hb in hr.boxes.xyxy.tolist():
                m = [j for j, q in enumerate(heads) if iou(q, hb) > 0.25]
                if m:
                    heads[m[0]] = hb
                else:
                    heads.append(hb)
            keep = []
            for h in sorted(heads, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
                if all(iou(h, k) < 0.4 for k in keep):
                    keep.append(h)
            out.append(keep)
    return [[[v / s for v in h] for h in hs] for hs, (_, s) in zip(out, ups)]


def patch_of(frame, hb, P=128):
    H, W = frame.shape[:2]
    cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
    half = max(hb[2] - hb[0], hb[3] - hb[1])
    x1, y1, x2, y2 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    pad = [max(0, -y1), max(0, y2 - H), max(0, -x1), max(0, x2 - W)]
    p = cv2.copyMakeBorder(frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)], *pad, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(p, (P, P), interpolation=cv2.INTER_CUBIC)


def combine(obs):
    tot, wsum = {c: 0.0 for c in C}, 0.0
    for px, p in obs:
        w = min(px, 40) / 40
        for c in C:
            tot[c] += w * math.log(max(p[c], 1e-6))
        wsum += w
    z = {c: math.exp(tot[c] / wsum) for c in C}
    n = sum(z.values())
    return {c: z[c] / n for c in C}


def decide(obs):
    """helmet / no_helmet / not_sure for one person's frames, or None if it is not a rider head."""
    p = combine(obs)
    c = max(C, key=lambda c: p[c])
    if c == "not_head":
        return None
    if p[c] < op["thr"] or (c == "no_helmet" and (p[c] < op["nh_thr"] or len(obs) < op.get("nh_min_frames", 1))):
        return "not_sure"
    # frames must agree: a person whose confident frames split between helmet and no_helmet is usually two
    # neighbouring riders merged by the tracker, so neither answer can be trusted
    other = "no_helmet" if c == "helmet" else "helmet"
    conf = [(min(px, 40) / 40, max(("helmet", "no_helmet"), key=lambda k: q[k])) for px, q in obs if max(q["helmet"], q["no_helmet"]) >= 0.7]
    wsum = sum(w for w, _ in conf)
    if wsum and sum(w for w, k in conf if k == other) / wsum >= op.get("max_disagree", 0.15):
        return "not_sure"
    # and the clearest (largest) view must not confidently say the opposite
    px, q = max(obs, key=lambda o: o[0])
    if q[other] >= 0.7:
        return "not_sure"
    return c


vp = Path(args.video)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(vp))
if not vp.exists() or not cap.isOpened() or not cap.read()[0]:
    sys.exit(f"cannot read video: {vp}")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
step = max(1, round(fps / args.fps))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.max_sec:
    n = min(n, int(args.max_sec * fps))
W, H = int(cap.get(3)), int(cap.get(4))
writer = cv2.VideoWriter(str(out / f"{vp.stem}_helmet.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (W, H))

# person key = (track, slot cluster by relative x); obs = [(head_px, probs)]
persons = defaultdict(list)
slots = defaultdict(list)  # track -> [rel_x of known persons]
seen_span = {}  # person -> [first t, first centre, first px, last t, last centre, last px]
best_view = {}  # person -> (head_px, patch, time s) of the clearest view, for the review sheet
moto_seen, moto_checked = set(), set()
moto_start, moto_moved = {}, set()  # first centre per motorcycle track / tracks that have moved  # motorcycle tracks seen / with at least one usable head
fi = -1
while True:
    ok = cap.grab()
    fi += 1
    if not ok or fi >= n:
        break
    if fi % step:
        continue
    ok, f = cap.retrieve()
    r = idd.track(f, imgsz=960, conf=0.3, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    draws = []
    if r.boxes.id is not None:
        boxes = list(zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()))
        # low-confidence "motorcycles" are where false scenes come from (e.g. a TV news desk read as handlebars)
        # second opinion: an independently trained detector (COCO) must also see a motorcycle there
        coco_m = coco.predict(f, imgsz=960, conf=0.25, classes=[3], verbose=False)[0].boxes.xyxy.tolist()
        motos = [(b, t) for b, t, c, cf in boxes if r.names[c] == "motorcycle" and cf >= MOTO_CONF
                 and any(max(overlap(b, m), overlap(m, b)) > 0.5 for m in coco_m)]
        for b, t, c, cf in boxes:
            if r.names[c] == "motorcycle":
                moto_seen.add(t)
        riders_ = [b for b, t, c, cf in boxes if r.names[c] == "rider"]
        jobs = []
        # helmets are only required on motorised two-wheelers: bicycles, cycle-rickshaws and rider
        # boxes that are not on a motorcycle are skipped
        for b, tid in motos:
            moto_seen.add(tid)
            box = b[:]
            ridden = False
            for rd in riders_:
                if overlap(b, rd) > 0.2 or overlap(rd, b) > 0.2:
                    box = [min(box[0], rd[0]), min(box[1], rd[1]), max(box[2], rd[2]), max(box[3], rd[3])]
                    ridden = True
            # only check bikes someone is on: a rider box on the bike, or the bike has moved (parked bikes
            # otherwise let wall murals, posters and bystanders behind them through as "riders")
            c0 = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            start = moto_start.setdefault(tid, c0)
            if math.dist(start, c0) > 0.5 * (b[2] - b[0]):
                moto_moved.add(tid)
            if not ridden and tid not in moto_moved:
                continue
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bh < 40:
                continue
            up = 0.25 if box[1] < b[1] - 0.1 * (b[3] - b[1]) else 0.6  # rider box already covers the heads
            x1, y1, x2, y2 = map(int, [max(0, box[0] - 0.2 * bw), max(0, box[1] - up * bh), min(W, box[2] + 0.2 * bw), min(H, box[3] + 0.05 * bh)])
            if x2 - x1 > 4 and y2 - y1 > 4:
                jobs.append((box, tid, x1, y1, f[y1:y2, x1:x2]))
        items = []
        for (b, tid, x1, y1, _), hs in zip(jobs, heads_batch([j[4] for j in jobs]) if jobs else []):
            bw, bh = b[2] - b[0], b[3] - b[1]
            for h in hs:
                h = [h[0] + x1, h[1] + y1, h[2] + x1, h[3] + y1]
                if min(h[2] - h[0], h[3] - h[1]) < args.min_head or (h[1] + h[3]) / 2 > b[1] + 0.6 * bh:
                    continue
                # a rider's head sits over the bike: horizontally inside the vehicle/rider box and not far above
                # it (rejects wall murals, posters, shop signs and people standing behind parked bikes)
                hcx, hs_ = (h[0] + h[2]) / 2, max(h[2] - h[0], h[3] - h[1])
                if not b[0] - 0.1 * bw <= hcx <= b[2] + 0.1 * bw or h[3] < b[1] - 0.5 * bh or hs_ > 0.6 * bw + 0.4 * bh:
                    continue
                rel = ((h[0] + h[2]) / 2 - b[0]) / bw
                known = slots[tid]
                j = min(range(len(known)), key=lambda k: abs(known[k] - rel)) if known else None
                if j is None or abs(known[j] - rel) > 0.18:
                    known.append(rel)
                    j = len(known) - 1
                items.append(((tid, j), h))
                moto_checked.add(tid)
        if items:
            probs = cls.predict([patch_of(f, h) for _, h in items], imgsz=128, verbose=False)
            for (key, h), pr in zip(items, probs):
                p = {pr.names[k]: float(v) for k, v in enumerate(pr.probs.data.tolist())}
                px = min(h[2] - h[0], h[3] - h[1])
                persons[key].append((px, p))
                t_now, c_now = fi / fps, ((h[0] + h[2]) / 2, (h[1] + h[3]) / 2)
                if key not in seen_span:
                    seen_span[key] = [t_now, c_now, px, t_now, c_now, px]
                seen_span[key][3:] = [t_now, c_now, px]
                if px > best_view.get(key, (0,))[0]:
                    best_view[key] = (px, patch_of(f, h), round(fi / fps, 1), [round(v, 1) for v in h])
                draws.append((key, h))
    # draw with the decision so far (the decision firms up as frames accumulate)
    for key, h in draws:
        d = decide(persons[key])
        if d is None:
            continue
        x1, y1, x2, y2 = map(int, h)
        cv2.rectangle(f, (x1, y1), (x2, y2), COLORS[d], 2)
        cv2.putText(f, d.replace("_", " "), (x1, max(12, y1 - 4)), 0, 0.5, COLORS[d], 2)
    writer.write(f)
writer.release()

# stitch tracks: a person whose track starts within STITCH_S of where another person's track ended
# (close in position, similar head size) is the same rider re-acquired by the tracker
STITCH_S = 1.5
order_keys = sorted(persons, key=lambda k: seen_span[k][0])
parent = {k: k for k in order_keys}
def root(k):
    while parent[k] != k:
        k = parent[k]
    return k
for i, b in enumerate(order_keys):
    fb = seen_span[b]
    best = None
    for a in order_keys[:i]:
        la = seen_span[a]
        gap = fb[0] - la[3]
        if not 0 < gap <= STITCH_S or root(a) == root(b):
            continue
        dist = math.dist(la[4], fb[1]) / max(la[5], fb[2])
        if dist < 1.5 + 2.0 * gap and 0.5 < fb[2] / max(la[5], 1e-6) < 2.0:
            if best is None or dist < best[0]:
                best = (dist, a)
    if best:
        parent[b] = root(best[1])
merged = defaultdict(list)
for k in order_keys:
    merged[root(k)].extend(persons[k])
    if best_view[k][0] > best_view[root(k)][0]:
        best_view[root(k)] = best_view[k]
n_raw = len(persons)
persons = merged

riders = []
for key, obs in persons.items():
    p = combine(obs)
    d = decide(obs)
    if d is None:
        continue
    riders.append({"track": key[0], "slot": key[1], "decision": d, "frames": len(obs), "t_best": best_view[key][2], "head_box_best": best_view[key][3],
                   "max_head_px": round(max(o[0] for o in obs), 1), "p": {k: round(v, 3) for k, v in p.items()},
                   "obs": [[round(px, 1)] + [round(q[c], 3) for c in C] for px, q in obs]})
# review sheet: every rider's clearest head, framed in the decision colour, sorted no_helmet / not_sure / helmet
order = {"no_helmet": 0, "not_sure": 1, "helmet": 2}
keys = sorted([(r["track"], r["slot"]) for r in riders], key=lambda k: order[decide(persons[k])])
if keys:
    cells = []
    for k in keys:
        im = best_view[k][1].copy()
        d = decide(persons[k])
        cv2.rectangle(im, (0, 0), (127, 127), COLORS[d], 4)
        cv2.putText(im, f"{d.replace('_', ' ')} {best_view[k][2]}s", (4, 122), 0, 0.38, COLORS[d], 1)
        cells.append(im)
    cells += [np.full_like(cells[0], 255)] * (-len(cells) % 8)
    cv2.imwrite(str(out / f"{vp.stem}_review.jpg"), np.vstack([np.hstack(cells[i:i + 8]) for i in range(0, len(cells), 8)]))

summary = {"video": vp.name, "riders": len(riders), "raw_person_tracks": n_raw,
           **{k: sum(1 for x in riders if x["decision"] == k) for k in ("helmet", "no_helmet", "not_sure")}}
summary["not_sure_rate"] = round(summary["not_sure"] / max(1, len(riders)), 3)
# motorcycles the camera saw but never checked: parked / no rider seen, or heads never big enough (e.g. drone or
# very high CCTV views) - reported separately, never counted as helmet or no_helmet
summary["motorcycles_not_checked"] = len(moto_seen - moto_checked)  # parked, no visible rider, or heads too small
(out / f"{vp.stem}_riders.json").write_text(json.dumps({"summary": summary, "operating_point": op, "riders": riders}, indent=1))
print(json.dumps(summary))
