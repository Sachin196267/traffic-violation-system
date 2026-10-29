#!/usr/bin/env python3
"""
Stage 5 - one decision per motorcycle track from the out-of-fold crop probabilities, and the
operating point.

Per track the crop probabilities are combined (size-weighted geometric mean).  Decision:
    not_bike    most likely not a two-wheeler -> no decision
    TRIPLE      p(three_plus) >= t_hi and at least --min-votes crops individually say three_plus
    not triple  p(three_plus) <= t_lo
    not sure    otherwise
The thresholds are chosen on held-out-video predictions only:
    1. zero wrong TRIPLE calls (else as few as possible),
    2. "not sure" at most --max-not-sure of the two-wheeler tracks (~1 in 30),
    3. then as many real triples caught as possible.
Writes runs/triple_rider/operating_point.json and prints the confusion tables.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
C = ("one", "two", "three_plus", "not_bike")
ap = argparse.ArgumentParser()
ap.add_argument("--max-not-sure", type=float, default=1 / 30)
ap.add_argument("--min-votes", type=int, default=1)
ap.add_argument("--out", default=str(ROOT.parent / "runs/triple_rider/operating_point.json"))
a = ap.parse_args()

samples = json.loads((ROOT / "samples.json").read_text())
oof = json.loads((ROOT / "oof.json").read_text())
tracks = defaultdict(list)
truth = {}
for s in samples:
    if s["path"] in oof:
        tracks[s["key"]].append((s["crop_h"], oof[s["path"]]))
        truth[s["key"]] = s["label"]


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


T = {}
for k, obs in tracks.items():
    p = combine(obs)
    votes = sum(max(C, key=lambda c: q[c]) == "three_plus" for _, q in obs)
    T[k] = (p, votes, len(obs))


def decide(p, votes, t_lo, t_hi, min_votes):
    if max(C, key=lambda c: p[c]) == "not_bike":
        return "not_bike"
    if p["three_plus"] >= t_hi and votes >= min_votes:
        return "triple"
    if p["three_plus"] <= t_lo:
        return "not_triple"
    return "not_sure"


def table(t_lo, t_hi, mv):
    res = defaultdict(lambda: defaultdict(int))
    for k, (p, v, n) in T.items():
        res[truth[k]][decide(p, v, t_lo, t_hi, mv)] += 1
    return res


def stats(res):
    bikes = sum(res[c][d] for c in ("one", "two", "three_plus") for d in ("triple", "not_triple", "not_sure"))
    ns = sum(res[c]["not_sure"] for c in ("one", "two", "three_plus"))
    fp = res["one"]["triple"] + res["two"]["triple"] + res["not_bike"]["triple"]
    tp = res["three_plus"]["triple"]
    pos = sum(res["three_plus"].values())
    return dict(bike_tracks=bikes, not_sure=ns / max(bikes, 1), wrong_triple_calls=fp, triples_caught=tp,
                triples_total=pos, triple_recall=tp / max(pos, 1), precision=tp / max(tp + fp, 1))


best = None
for mv in sorted({1, a.min_votes}):
    for t_hi in np.arange(0.30, 0.995, 0.01):
        for t_lo in np.arange(0.01, t_hi, 0.01):
            s = stats(table(t_lo, t_hi, mv))
            if s["not_sure"] > a.max_not_sure:
                continue
            key = (-s["wrong_triple_calls"], s["triples_caught"], -s["not_sure"])
            if best is None or key > best[0]:
                best = (key, float(t_lo), float(t_hi), mv, s)
_, t_lo, t_hi, mv, s = best
res = table(t_lo, t_hi, mv)
print(f"operating point: t_lo={t_lo:.2f} t_hi={t_hi:.2f} min_votes={mv}")
print(json.dumps(s, indent=1))
print(f"{'truth':12s} " + " ".join(f"{d:>11s}" for d in ("triple", "not_triple", "not_sure", "not_bike")))
for c in C:
    print(f"{c:12s} " + " ".join(f"{res[c][d]:11d}" for d in ("triple", "not_triple", "not_sure", "not_bike")))
wrong = [(k, round(T[k][0]['three_plus'], 3)) for k in T
         if truth[k] != "three_plus" and decide(*T[k][:2], t_lo, t_hi, mv) == "triple"]
missed = [(k, round(T[k][0]['three_plus'], 3)) for k in T
          if truth[k] == "three_plus" and decide(*T[k][:2], t_lo, t_hi, mv) != "triple"]
print("wrong triple calls:", wrong)
print("triples not called:", missed)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(dict(t_lo=round(t_lo, 3), t_hi=round(t_hi, 3), min_votes=mv,
                                       target_not_sure=a.max_not_sure,
                                       chosen_on="leave-videos-out out-of-fold predictions", cv=s), indent=1))
