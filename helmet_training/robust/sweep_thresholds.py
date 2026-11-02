#!/usr/bin/env python3
"""
Precision-first operating points: for each (thr, nh_thr) pair, person-level results on the
out-of-fold predictions (oof.json) - how many riders become "not sure" vs how many wrong calls remain.

  python sweep_thresholds.py [--oof oof.json]
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--oof", default=str(ROOT / "oof.json"))
args = ap.parse_args()

samples = json.loads((ROOT / "samples.json").read_text())
oof = json.loads(Path(args.oof).read_text())
C = ("helmet", "no_helmet", "not_head")
persons = defaultdict(list)
for s in samples:
    if s["path"] in oof:
        persons[(s["video"], s["pid"])].append(s)


def combine(ss):
    tot, wsum = {c: 0.0 for c in C}, 0.0
    for s in ss:
        w = min(s["head_px"], 40) / 40
        for c in C:
            tot[c] += w * math.log(max(oof[s["path"]][c], 1e-6))
        wsum += w
    z = {c: math.exp(tot[c] / wsum) for c in C}
    n = sum(z.values())
    return {c: z[c] / n for c in C}


P = [(ss[0]["label"], combine(ss), len(ss)) for ss in persons.values() if ss[0]["label"] != "x"]


def run(thr, nh_thr, min_frames=1):
    ns = wrong_ch = missed = ok = forced_u = nh_caught = 0
    n_h = sum(1 for l, *_ in P if l == "h")
    n_n = sum(1 for l, *_ in P if l == "n")
    for l, p, nf in P:
        c = max(C, key=lambda c: p[c])
        if c == "not_head":
            continue
        if p[c] < thr or (c == "no_helmet" and (p[c] < nh_thr or nf < min_frames)):
            ns += 1
            continue
        if l == "u":
            forced_u += 1
        elif (l == "h") == (c == "helmet"):
            ok += 1
            nh_caught += c == "no_helmet"
        elif c == "no_helmet":
            wrong_ch += 1
        else:
            missed += 1
    called = ok + wrong_ch + missed
    return {"not_sure": ns / len(P), "wrong_calls": (wrong_ch + missed) / max(1, called), "wrong_challans": wrong_ch,
            "of_helmeted": n_h, "missed": missed, "nh_caught": nh_caught, "of_nh": n_n, "forced_u": forced_u}


print(f"{'thr':>5} {'nh_thr':>6} | {'not sure':>8} | {'wrong calls':>11} | wrong challans | missed nh | nh caught | unclear-u forced")
for thr in (0.55, 0.7, 0.8, 0.9, 0.95):
    for nh_thr in sorted({max(thr, x) for x in (0.8, 0.9, 0.95, 0.98)}):
        r = run(thr, nh_thr)
        print(f"{thr:5.2f} {nh_thr:6.2f} | {r['not_sure']:8.1%} | {r['wrong_calls']:11.1%} | {r['wrong_challans']:4d}/{r['of_helmeted']} "
              f"({r['wrong_challans'] / r['of_helmeted']:.1%}) | {r['missed']:9d} | {r['nh_caught']:3d}/{r['of_nh']} | {r['forced_u']}")

print("\nchosen precision-first setting (thr 0.80, nh_thr 0.95, no_helmet needs >= 3 frames):")
r = run(0.8, 0.95, 3)
print(f"  not sure {r['not_sure']:.1%} | wrong calls {r['wrong_calls']:.1%} | wrong challans {r['wrong_challans']}/{r['of_helmeted']} | "
      f"missed no_helmet {r['missed']} | no_helmet caught {r['nh_caught']}/{r['of_nh']} | unclear-u forced {r['forced_u']}")
