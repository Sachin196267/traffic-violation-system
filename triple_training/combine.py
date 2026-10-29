#!/usr/bin/env python3
"""
Step 4 - combine the classifier with the heads-on-seat count into one explainable score per bike,
and choose the operating point.

Per motorcycle (track) features, from all its crops:
    lp3   log-odds of the size-weighted p(three_plus)            (classifier)
    lp2   log-odds of p(two)
    h3    share of crops with >= 3 heads on this bike's seat     (head counter)
    hmax  most heads seen on the seat in any crop (capped at 4)
A small logistic model turns these into P(triple).  It is fitted LEAVE-ONE-FOLD-OUT (each fold is
scored by a model fitted on the other folds), so every score below is for unseen cameras.
The final combiner (all folds) is saved for run_video.py.

Thresholds: TRIPLE if P >= t_hi (chosen for 0 wrong TRIPLE calls on held-out bikes), not triple if
P <= t_lo, "not sure" in between (at most --max-not-sure of two-wheelers).
Writes runs/triple_rider/operating_point.json and prints both "classifier only" and "combined".
"""
import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
C = ("one", "two", "three_plus", "not_bike")
ap = argparse.ArgumentParser()
ap.add_argument("--max-not-sure", type=float, default=1 / 12)
ap.add_argument("--out", default=str(R.parent / "runs/triple_rider/operating_point.json"))
a = ap.parse_args()

samples = json.loads((R / "samples.json").read_text())
oof = json.loads((R / "oof.json").read_text())
heads = {}
for f in glob.glob(str(R / "data/*/heads_seat.json")):
    heads.update(json.loads(Path(f).read_text()))


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


def logit(p):
    p = min(max(p, 1e-5), 1 - 1e-5)
    return math.log(p / (1 - p))


def features(p, hs):
    hs = hs or [0]
    return [logit(p["three_plus"]), logit(p["two"]), float(np.mean([h >= 3 for h in hs])), min(max(hs), 4) / 4]


tracks = defaultdict(list)
info = {}
for s in samples:
    if s["path"] not in oof or s["fold"] < 0:
        continue
    h = heads.get(Path(s["path"]).name, {}).get("n")
    tracks[s["key"]].append((s["crop_h"], oof[s["path"]], h))
    info[s["key"]] = (s["label"], s["fold"])
keys = sorted(tracks)
P = {k: combine([(px, q) for px, q, _ in tracks[k]]) for k in keys}
X = np.array([features(P[k], [h for _, _, h in tracks[k] if h is not None]) for k in keys])
y = np.array([info[k][0] == "three_plus" for k in keys], float)
fold = np.array([info[k][1] for k in keys])
is_bike = np.array([max(C, key=lambda c: P[k][c]) != "not_bike" for k in keys])
label = [info[k][0] for k in keys]


def fit(Xt, yt, l2=1.0, iters=3000, lr=0.1):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Z = (Xt - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    pos_w = (len(yt) - yt.sum()) / max(yt.sum(), 1)            # balance the rare positives
    sw = np.where(yt == 1, pos_w, 1.0)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = sw * (p - yt)
        w -= lr * (Z.T @ g / len(yt) + l2 * w / len(yt))
        b -= lr * g.mean()
    return dict(mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(), b=float(b))


def predict(m, Xs):
    Z = (Xs - np.array(m["mu"])) / np.array(m["sd"])
    return 1 / (1 + np.exp(-(Z @ np.array(m["w"]) + m["b"])))


score_comb = np.zeros(len(keys))
for f in sorted(set(fold)):
    tr = fold != f
    score_comb[fold == f] = predict(fit(X[tr], y[tr]), X[fold == f])
score_cls = np.array([P[k]["three_plus"] for k in keys])
votes = np.array([sum(q["three_plus"] >= 0.5 for _, q, _ in tracks[k]) for k in keys])
nframes = np.array([len(tracks[k]) for k in keys])


def operating_point(score, mv=1):
    bikes = is_bike
    neg = bikes & (y == 0)
    pos = y == 1
    ok = votes >= mv                                            # enough frames individually agree
    best = None
    for t_hi in np.unique(np.r_[score[pos], score[neg]]):
        if (neg & ok & (score >= t_hi)).sum() > 0:
            continue                                            # must make 0 wrong TRIPLE calls
        caught = int((pos & bikes & ok & (score >= t_hi)).sum())
        # widest "not sure" band allowed: pick t_lo as low as the not-sure budget permits
        forced = int((bikes & (score >= t_hi) & ~ok).sum())   # high score, too few agreeing frames -> not sure
        below = np.sort(score[bikes & (score < t_hi)])[::-1]
        budget = int(a.max_not_sure * bikes.sum()) - forced
        if budget < 0:
            continue
        t_lo = below[budget] if budget < len(below) else -1.0
        key = (caught, -t_hi)
        if best is None or key > best[0]:
            best = (key, float(t_hi), float(t_lo))
    _, t_hi, t_lo = best
    trip = (score >= t_hi) & ok
    dec = np.where(~bikes, "not_bike", np.where(trip, "triple", np.where(score > t_lo, "not_sure", "not_triple")))
    return t_hi, t_lo, dec


res = {}
for name, sc, mv in [(f"classifier, >={m} agreeing frames", score_cls, m) for m in (1, 2, 3, 4)] + \
                     [(f"classifier + heads, >={m} agreeing frames", score_comb, m) for m in (1, 3)]:
    t_hi, t_lo, dec = operating_point(sc, mv)
    tab = defaultdict(lambda: defaultdict(int))
    for l, d in zip(label, dec):
        tab[l][d] += 1
    nb = int(is_bike.sum())
    tp = tab["three_plus"]["triple"]
    stats = dict(t_hi=round(t_hi, 5), t_lo=round(t_lo, 5), min_votes=mv, bikes=nb, triples=int(y.sum()),
                 caught=tp, wrong=sum(tab[l]["triple"] for l in ("one", "two", "not_bike")),
                 not_sure=round(sum(tab[l]["not_sure"] for l in C) / nb, 3),
                 triples_not_sure=tab["three_plus"]["not_sure"], triples_missed=tab["three_plus"]["not_triple"])
    res[name] = stats
    print(f"\n== {name}: {json.dumps(stats)}")
    print(f"{'truth':11s}" + "".join(f"{d:>11s}" for d in ("triple", "not_triple", "not_sure", "not_bike")))
    for l in C:
        print(f"{l:11s}" + "".join(f"{tab[l][d]:11d}" for d in ("triple", "not_triple", "not_sure", "not_bike")))

use = max(res, key=lambda n: (res[n]["wrong"] == 0, res[n]["not_sure"] <= a.max_not_sure + 1e-9,
                                res[n]["caught"], -res[n]["min_votes"]))
final = fit(X, y)
out = dict(method=use, combiner=final if "heads" in use else None,
           features=["logit p_triple", "logit p_two", "share of frames with >=3 heads on seat", "max heads on seat / 4"],
           t_hi=res[use]["t_hi"], t_lo=res[use]["t_lo"], min_votes=res[use]["min_votes"],
           target="0 wrong TRIPLE calls on held-out cameras; not-sure <= %.3f of two-wheelers" % a.max_not_sure,
           cv=res)
Path(a.out).write_text(json.dumps(out, indent=1))
print("\nchosen:", use, "->", a.out)
