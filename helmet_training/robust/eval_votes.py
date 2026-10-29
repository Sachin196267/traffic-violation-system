#!/usr/bin/env python3
"""
Person-level evaluation on out-of-fold predictions (every video is scored by a model that never saw it).

Per person, frame probabilities are combined (size-weighted mean of log-probs). A person is
  * dropped   if not_head wins (not a rider head),
  * NOT SURE  if the winning class' combined probability < threshold,
  * helmet / no_helmet otherwise; no_helmet additionally needs p >= nh_threshold (wrong challans cost more).
The threshold is picked so that "not sure" hits the target rate (default 1 in 30 labelled rider persons).
Also reports single-frame (no voting) numbers for comparison.
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--target", type=float, default=1 / 30)
ap.add_argument("--max-frames", type=int, default=0, help="use only the first N frames per person (0 = all)")
ap.add_argument("--json", help="write summary json here")
args = ap.parse_args()

samples = json.loads((ROOT / "samples.json").read_text())
oof = json.loads((ROOT / "oof.json").read_text())
C = ("helmet", "no_helmet", "not_head")
persons = defaultdict(list)
for s in samples:
    if s["path"] in oof:
        persons[(s["video"], s["pid"])].append(s)


def combine(ss):
    ss = sorted(ss, key=lambda s: s["path"])[: args.max_frames or None]
    tot = {c: 0.0 for c in C}
    wsum = 0
    for s in ss:
        w = min(s["head_px"], 40) / 40
        for c in C:
            tot[c] += w * math.log(max(oof[s["path"]][c], 1e-6))
        wsum += w
    z = {c: math.exp(tot[c] / wsum) for c in C}
    n = sum(z.values())
    return {c: z[c] / n for c in C}


P = []
for key, ss in persons.items():
    lab = ss[0]["label"]
    if lab == "x":
        continue
    P.append({"video": key[0], "label": lab, "p": combine(ss), "n": len(ss), "px": max(s["head_px"] for s in ss)})


def decide(p, thr, nh_thr):
    c = max(C, key=lambda c: p[c])
    if c == "not_head":
        return "dropped"
    if p[c] < thr or (c == "no_helmet" and p[c] < nh_thr):
        return "not_sure"
    return c


def score(thr, nh_thr, rows):
    d = [(r, decide(r["p"], thr, nh_thr)) for r in rows]
    riders = [x for x in d]  # all labelled rider persons incl. "u"
    ns = sum(1 for _, o in riders if o == "not_sure") / max(1, len(riders))
    hn = [(r["label"], o) for r, o in d if r["label"] in "hn"]
    called = [(l, o) for l, o in hn if o in ("helmet", "no_helmet")]
    acc = sum(1 for l, o in called if (l == "h") == (o == "helmet")) / max(1, len(called))
    nh = [(l, o) for l, o in hn if l == "n"]
    h = [(l, o) for l, o in hn if l == "h"]
    return {"persons": len(riders), "not_sure": round(ns, 4), "accuracy_called": round(acc, 4),
            "no_helmet_caught": f"{sum(1 for _, o in nh if o == 'no_helmet')}/{len(nh)}",
            "no_helmet_recall": round(sum(1 for _, o in nh if o == "no_helmet") / max(1, len(nh)), 4),
            "wrong_challans(helmet->no_helmet)": f"{sum(1 for _, o in h if o == 'no_helmet')}/{len(h)}",
            "missed(no_helmet->helmet)": sum(1 for _, o in nh if o == "helmet"),
            "dropped_as_not_head": sum(1 for _, o in d if o == "dropped"),
            "unclear_u_forced_calls": sum(1 for r, o in d if r["label"] == "u" and o in ("helmet", "no_helmet"))}


# pick nh_thr then thr: sweep, keep not_sure <= target, maximise accuracy
best = None
for nh_thr in [0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
    lo, hi = 0.34, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if score(mid, nh_thr, P)["not_sure"] > args.target:
            hi = mid
        else:
            lo = mid
    s = score(lo, nh_thr, P)
    # only settings that meet the not-sure target are eligible; among them, fewest wrong challans per no_helmet caught
    feasible = s["not_sure"] <= args.target + 1e-9
    key = (feasible, s["accuracy_called"], s["no_helmet_recall"])
    print(f"nh_thr {nh_thr}: thr {lo:.3f} {'ok ' if feasible else 'OVER'} {s}")
    if best is None or key > best[0]:
        best = (key, lo, nh_thr, s)
_, thr, nh_thr, s = best
print("\nCHOSEN thr", round(thr, 3), "nh_thr", nh_thr)
print("ALL:", s)
per_video = {}
for v in sorted({r["video"] for r in P}):
    per_video[v] = score(thr, nh_thr, [r for r in P if r["video"] == v])
    print(f"  {v:14s}", per_video[v])

# single-frame baseline (each frame judged alone, same thresholds)
single = []
for s_ in samples:
    if s_["path"] in oof and s_["label"] in "hnu":
        single.append({"video": s_["video"], "label": s_["label"], "p": oof[s_["path"]]})
print("\nSINGLE FRAME (no voting):", score(thr, nh_thr, single))
if args.json:
    Path(args.json).write_text(json.dumps({"thr": thr, "nh_thr": nh_thr, "all": s, "per_video": per_video}, indent=1))
