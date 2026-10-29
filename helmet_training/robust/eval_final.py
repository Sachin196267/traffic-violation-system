#!/usr/bin/env python3
"""
Person-level held-out results with the exact decision rule of run_video.py (thresholds, min frames,
frame agreement, clearest-view check) on out-of-fold predictions.

  python eval_final.py oof.json [oof_aug.json ...]
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
C = ("helmet", "no_helmet", "not_head")
op = json.loads((ROOT.parents[1] / "runs/helmet_robust/final/operating_point.json").read_text())
samples = json.loads((ROOT / "samples.json").read_text())


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


def decide(obs):  # keep identical to run_video.py
    p = combine(obs)
    c = max(C, key=lambda c: p[c])
    if c == "not_head":
        return None
    if p[c] < op["thr"] or (c == "no_helmet" and (p[c] < op["nh_thr"] or len(obs) < op.get("nh_min_frames", 1))):
        return "not_sure"
    other = "no_helmet" if c == "helmet" else "helmet"
    conf = [(min(px, 40) / 40, max(("helmet", "no_helmet"), key=lambda k: q[k])) for px, q in obs if max(q["helmet"], q["no_helmet"]) >= 0.7]
    wsum = sum(w for w, _ in conf)
    if wsum and sum(w for w, k in conf if k == other) / wsum >= op.get("max_disagree", 0.15):
        return "not_sure"
    px, q = max(obs, key=lambda o: o[0])
    if q[other] >= 0.7:
        return "not_sure"
    return c


for f in sys.argv[1:]:
    oof = json.loads((ROOT / f).read_text())
    persons = defaultdict(list)
    for s in samples:
        if s["path"] in oof:
            persons[(s["video"], s["pid"])].append(s)
    n = defaultdict(int)
    for ss in persons.values():
        l = ss[0]["label"]
        if l == "x":
            continue
        d = decide([(s["head_px"], oof[s["path"]]) for s in ss])
        n["persons"] += 1
        n[f"{l}->{d}"] += 1
    H = sum(v for k, v in n.items() if k.startswith("h->"))
    N = sum(v for k, v in n.items() if k.startswith("n->"))
    ns = sum(v for k, v in n.items() if k.endswith("->not_sure"))
    called_hn = n["h->helmet"] + n["h->no_helmet"] + n["n->helmet"] + n["n->no_helmet"]
    wrong = n["h->no_helmet"] + n["n->helmet"]
    print(f"{f}: riders {n['persons']} | not sure {ns / n['persons']:.1%} | wrong calls {wrong}/{called_hn} ({wrong / called_hn:.1%}) | "
          f"wrong challans (helmet->no_helmet) {n['h->no_helmet']}/{H} | missed (no_helmet->helmet) {n['n->helmet']}/{N} | "
          f"no_helmet caught {n['n->no_helmet']}/{N} | helmet confirmed {n['h->helmet']}/{H} | unclear persons given an answer "
          f"{n['u->helmet'] + n['u->no_helmet']}/{sum(v for k, v in n.items() if k.startswith('u->'))}")
