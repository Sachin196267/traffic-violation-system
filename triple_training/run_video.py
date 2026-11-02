#!/usr/bin/env python3
"""
Triple-riding check on any video: one decision per motorcycle (TRIPLE / not triple / not sure).

    python triple_training/run_video.py <video> [--out dir] [--max-sec 120]

Same geometry as training: IDD detector + ByteTrack -> motorcycle tracks + riders sitting on
them -> letterboxed crop -> classifier (one / two / three_plus / not_bike) on every sampled
frame, plus heads counted on this bike's seat -> combined score (combine.py) -> TRIPLE only if the
score is high AND enough frames individually agree -> thresholds from
runs/triple_rider/operating_point.json (chosen on held-out cameras). Each bike in the JSON carries a
confidence, the evidence and plain-language reasons (for explainable AI).
Writes <out>/<video>_triple.mp4 and <out>/<video>_triple.json.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "triple_training"))
C = ("one", "two", "three_plus", "not_bike")
COL = {"triple": (0, 0, 255), "not_sure": (0, 170, 255), "not_triple": (0, 190, 0)}

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default=str(ROOT / "triple_training/video_out"))
ap.add_argument("--fps", type=float, default=5.0)
ap.add_argument("--max-sec", type=float, default=0)
ap.add_argument("--min-h", type=int, default=40)
ap.add_argument("--weights", default=str(ROOT / "runs/triple_rider/final/weights/last.pt"))
args = ap.parse_args()

op = json.loads((ROOT / "runs/triple_rider/operating_point.json").read_text())
idd = YOLO(str(ROOT / "runs/idd_final/weights/best.pt"))
cls = YOLO(args.weights)
from focus import letterbox_geom  # noqa: E402
from heads_seat import heads_in, on_seat  # noqa: E402  (pose + head models, same as training)


def on_bike(r, m):
    mw, mh = m[2] - m[0], m[3] - m[1]
    cx = (r[0] + r[2]) / 2
    return (m[0] - 0.15 * mw <= cx <= m[2] + 0.15 * mw and r[1] < m[1] + 0.3 * mh
            and m[1] - 0.2 * mh <= r[3] <= m[3] + 0.25 * mh)


def letterbox(img, size=256):
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                   interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    out = np.full((size, size, 3), 114, np.uint8)
    y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def combine(obs):
    tot, wsum = {c: 0.0 for c in C}, 0.0
    for px, p in obs:
        w = min(px, 160) / 160
        for c in C:
            tot[c] += w * math.log(max(p[c], 1e-6))
        wsum += w
    z = {c: math.exp(tot[c] / max(wsum, 1e-9)) for c in C}
    n = sum(z.values())
    return {c: z[c] / n for c in C}


def logit(v):
    v = min(max(v, 1e-5), 1 - 1e-5)
    return math.log(v / (1 - v))


def score_of(p, hs):
    """Final P(triple): classifier alone, or the logistic combiner with the head count (combine.py)."""
    m = op.get("combiner")
    if not m:
        return p["three_plus"]
    hs = hs or [0]
    x = np.array([logit(p["three_plus"]), logit(p["two"]), float(np.mean([h >= 3 for h in hs])), min(max(hs), 4) / 4])
    z = (x - np.array(m["mu"])) / np.array(m["sd"])
    return float(1 / (1 + np.exp(-(z @ np.array(m["w"]) + m["b"]))))


def decide(obs, hs):
    p = combine(obs)
    votes = sum(q["three_plus"] >= 0.5 for _, q in obs)
    if max(C, key=lambda c: p[c]) == "not_bike":
        return "not_bike", p, 0.0, votes
    sc = score_of(p, hs)
    if sc >= op["t_hi"] and votes >= op["min_votes"]:
        return "triple", p, sc, votes
    if sc <= op["t_lo"]:
        return "not_triple", p, sc, votes
    return "not_sure", p, sc, votes


vp = Path(args.video)
cap = cv2.VideoCapture(str(vp))
if not cap.isOpened() or not cap.read()[0]:
    sys.exit(f"cannot read video: {vp}")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
step = max(1, round(fps / args.fps))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.max_sec:
    n = min(n, int(args.max_sec * fps))
W, H = int(cap.get(3)), int(cap.get(4))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
scale = min(1.0, 1280 / W)
size = (round(W * scale), round(H * scale))
writer = cv2.VideoWriter(str(out / f"{vp.stem}_triple.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps / step, size)
if not writer.isOpened():
    writer = cv2.VideoWriter(str(out / f"{vp.stem}_triple.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, size)

obs = defaultdict(list)
heads = defaultdict(list)          # heads on this bike's seat, per sampled frame
first_seen, last_box = {}, {}
fi = -1
while True:
    ok = cap.grab()
    fi += 1
    if not ok or fi >= n:
        break
    if fi % step:
        continue
    ok, f = cap.retrieve()
    if not ok:
        break
    r = idd.track(f, imgsz=960, conf=0.3, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    shown = []
    if r.boxes.id is not None:
        boxes = r.boxes.xyxy.tolist()
        names = [r.names[c] for c in r.boxes.cls.int().tolist()]
        people = [b for b, nm in zip(boxes, names) if nm in ("rider", "person")]
        jobs = []
        for b, nm, tid in zip(boxes, names, r.boxes.id.int().tolist()):
            if nm != "motorcycle" or b[3] - b[1] < args.min_h:
                continue
            att = [p for p in people if on_bike(p, b)]
            u = [min([b[0]] + [p[0] for p in att]), min([b[1]] + [p[1] for p in att]),
                 max([b[2]] + [p[2] for p in att]), max([b[3]] + [p[3] for p in att])]
            uw, uh = u[2] - u[0], u[3] - u[1]
            if not att:
                u[1] -= 0.9 * uh
                uh = u[3] - u[1]
            x1, y1 = int(max(0, u[0] - 0.12 * uw)), int(max(0, u[1] - 0.08 * uh))
            x2, y2 = int(min(W, u[2] + 0.12 * uw)), int(min(H, u[3] + 0.04 * uh))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                sc_, ox, oy = letterbox_geom(x2 - x1, y2 - y1)
                moto_lb = [(b[0] - x1) * sc_ + ox, (b[1] - y1) * sc_ + oy, (b[2] - x1) * sc_ + ox, (b[3] - y1) * sc_ + oy]
                jobs.append((tid, [x1, y1, x2, y2], letterbox(f[y1:y2, x1:x2]), moto_lb))
        if jobs and op.get("combiner"):
            for (tid, _, _, mlb), hs in zip(jobs, heads_in([j[2] for j in jobs])):
                heads[tid].append(len(on_seat(hs, mlb)))
        if jobs:
            for (tid, box, _, _), pr in zip(jobs, cls.predict([j[2] for j in jobs], imgsz=256, verbose=False)):
                obs[tid].append((box[3] - box[1], {pr.names[j]: float(v) for j, v in enumerate(pr.probs.data.tolist())}))
                first_seen.setdefault(tid, fi / fps)
                last_box[tid] = box
                shown.append((tid, box))
    vis = cv2.resize(f, size) if scale < 1 else f.copy()
    for tid, box in shown:
        d, p, _, _ = decide(obs[tid], heads[tid])
        if d == "not_bike":
            continue
        b = [int(v * scale) for v in box]
        col = COL[d]
        th = 3 if d != "not_triple" else 1
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), col, th)
        txt = {"triple": "TRIPLE RIDING", "not_sure": "not sure", "not_triple": ""}[d]
        if txt:
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (b[0], b[1] - tht - 8), (b[0] + tw + 6, b[1]), col, -1)
            cv2.putText(vis, txt, (b[0] + 3, b[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    writer.write(vis)
writer.release()
cap.release()

result = []
for tid, o in obs.items():
    d, p, sc, votes = decide(o, heads[tid])
    if d == "not_bike":
        continue
    # explainable output: how sure, and why
    conf = sc if d == "triple" else (1 - sc if d == "not_triple" else 0.5)
    hs = heads[tid]
    reasons = [f"classifier: P(3+ riders)={p['three_plus']:.2f}, P(2)={p['two']:.2f}, P(1)={p['one']:.2f}",
               f"{votes} of {len(o)} frames individually say 3+ riders (need {op['min_votes']})"]
    if hs:
        reasons.append(f"heads on this bike's seat: up to {max(hs)}, 3+ in {sum(h >= 3 for h in hs)} of {len(hs)} frames")
    if d == "not_sure":
        reasons.append("score between the 'not triple' and 'TRIPLE' limits"
                       if sc > op["t_lo"] and sc < op["t_hi"] else "high score but too few agreeing frames")
    result.append(dict(track=tid, decision=d, confidence=round(conf, 3), score=round(sc, 4),
                       p_triple=round(p["three_plus"], 3), p_two=round(p["two"], 3), p_one=round(p["one"], 3),
                       frames=len(o), frames_voting_triple=votes, heads_on_seat=hs, reasons=reasons,
                       thresholds=dict(triple_if_score_at_least=op["t_hi"], min_agreeing_frames=op["min_votes"],
                                       not_triple_if_score_at_most=op["t_lo"]),
                       first_seen_s=round(first_seen[tid], 1), last_box=last_box[tid]))
summary = {d: sum(r["decision"] == d for r in result) for d in ("triple", "not_triple", "not_sure")}
(out / f"{vp.stem}_triple.json").write_text(json.dumps(dict(video=vp.name, summary=summary, motorcycles=result), indent=1))
print(vp.stem, summary)
