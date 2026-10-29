#!/usr/bin/env python3
"""
Emergency-vehicle classifier = logistic regression on CLIP ViT-B/16 image embeddings.
Classes: ambulance / police / fire / other. Exempt from challans: ambulance, police, fire (user rule).

Training data
- web images (Indian / Pakistani emergency-vehicle datasets, CLIP pre-sorted on Kaggle): only images whose
  zero-shot class is confident (>= 0.9) are used, with that class as the label (their own dataset labels are wrong);
- vehicle crops from videos, labelled by eye per tracked vehicle (labels.json; unsure / junk left out).
Test: leave-videos-out (5 groups of videos). A vehicle's score = mean of its 3 highest crop scores.
Tracks from the videos that were not reviewed (zero-shot ambulance+police+fire < 0.1 in every crop; a random sample of 53 had
1 ambulance) count as 'other'.

    python ambulance_training/train_probe.py        -> runs/ambulance_cls/{probe.npz, operating_point.json, eval.json}
"""
import csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "runs/ambulance_cls"; OUT.mkdir(parents=True, exist_ok=True)
C = ["ambulance", "police", "fire", "other"]
ZS = ["ambulance", "police", "fire", "car", "van", "bus", "truck", "auto", "twowheeler", "other"]
LMAP = {"ambulance": "ambulance", "police": "police", "fire": "fire", "not": "other"}

# ---- web ----
rows = list(csv.DictReader(open(ROOT / "clip_web/clip_scores.csv")))
Ew = np.load(ROOT / "clip_web/clip_emb.npy").astype(np.float32)
Xw, yw = [], []
for r, e in zip(rows, Ew):
    p = {k: float(r[k]) for k in ZS}
    k = max(p, key=p.get)
    if p[k] < 0.9:
        continue
    Xw.append(e); yw.append(C.index(k) if k in C[:3] else 3)
Xw, yw = np.array(Xw), np.array(yw)
print("web training images:", {C[i]: int((yw == i).sum()) for i in range(4)})

# ---- videos ----
L = json.loads((ROOT / "labels.json").read_text())
V = {}   # video -> list of (track, emb (n,512), zero-shot ambulance probs, label or None)
for d in sorted((ROOT / "crops").iterdir()):
    if not (d / "clip.json").exists():
        continue
    j = json.loads((d / "clip.json").read_text()); m = j["meta"]
    E = np.load(d / "clip_emb.npy").astype(np.float32)
    per = defaultdict(list)
    for f, e in zip(j["files"], E):
        per[m[f]["track"]].append((e, m[f]["p"]))
    items = []
    for t, v in per.items():
        lab = L.get(f"vid:{d.name}:{t}")
        zs_amb = np.array([p["ambulance"] for _, p in v])
        zs_em_ = np.array([p["ambulance"] + p["police"] + p["fire"] for _, p in v])
        if lab is None:   # unreviewed: counted as 'other' only if no crop looks like ANY emergency type
            lab = "not" if zs_em_.max() < 0.1 else None
        if lab not in LMAP:
            continue                                               # unsure / junk / unreviewed middle
        zs_em = np.array([p["ambulance"] + p["police"] + p["fire"] for _, p in v])
        items.append((t, np.stack([e for e, _ in v]), zs_amb, zs_em, C.index(LMAP[lab])))
    V[d.name] = items
vids = sorted(V)
print("video tracks:", {C[i]: sum(1 for v in vids for it in V[v] if it[4] == i) for i in range(4)}, "videos", len(vids))

top3 = lambda a: float(np.mean(np.sort(a)[-3:]))
groups = [vids[i::5] for i in range(5)]
res = []
for g in groups:
    X, y = [Xw], [yw]
    for v in vids:
        if v in g:
            continue
        for t, E, _, _, lab in V[v]:
            X.append(E); y.append(np.full(len(E), lab))
    X, y = np.concatenate(X), np.concatenate(y)
    w = {i: len(y) / (4 * max(1, (y == i).sum())) for i in range(4)}
    clf = LogisticRegression(C=2.0, max_iter=3000, class_weight=w).fit(X, y)
    for v in g:
        for t, E, zs_amb, zs_em, lab in V[v]:
            P = clf.predict_proba(E)
            res.append({"video": v, "track": t, "label": C[lab],
                        "probe_amb": top3(P[:, 0]), "probe_em": top3(P[:, :3].sum(1)),
                        "probe_type": C[int(np.argmax(P[:, :3].mean(0)))],
                        "zs_amb": top3(zs_amb), "zs_em": top3(zs_em)})


def sweep(key, pos):
    s = np.array([r[key] for r in res]); y = np.array([r["label"] in pos for r in res])
    out = {}
    for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
        k = s >= t
        out[t] = {"caught": int((k & y).sum()), "of": int(y.sum()), "false": int((k & ~y).sum()), "of_neg": int((~y).sum())}
    return out


EM = ("ambulance", "police", "fire")
ev = {"emergency (probe)": sweep("probe_em", EM), "emergency (zero-shot)": sweep("zs_em", EM),
      "ambulance (probe)": sweep("probe_amb", ("ambulance",)), "ambulance (zero-shot)": sweep("zs_amb", ("ambulance",))}
for k, v in ev.items():
    print(k); [print(f"   t={t:<5} caught {r['caught']}/{r['of']}  false {r['false']}/{r['of_neg']}") for t, r in v.items()]

# operating point on the held-out results: exempt = highest recall with 0 false exemptions;
# review = lowest threshold whose false-alarm rate stays <= 2 % of ordinary vehicles
best = "emergency (probe)"
s = np.array([r["probe_em"] for r in res]); y = np.array([r["label"] in EM for r in res])
cands = np.unique(np.r_[s, 0.99])
t_ex = min((t for t in cands if not np.any((s >= t) & ~y)), default=0.99)
t_ex = float(max(t_ex, 0.8))
t_rv = float(min((t for t in cands if np.mean(s[~y] >= t) <= 0.02), default=0.5))
t_rv = min(t_rv, t_ex)
op = {"model": "CLIP ViT-B/16 + logistic probe (ambulance/police/fire/other)", "t_exempt": round(t_ex, 3),
      "t_review": round(t_rv, 3), "exempt_types": list(EM),
      "held_out": {"emergency_tracks": int(y.sum()), "other_tracks": int((~y).sum()),
                   "exempted_correct": int(((s >= t_ex) & y).sum()), "exempted_wrong": int(((s >= t_ex) & ~y).sum()),
                   "review_or_exempt_emergency": int(((s >= t_rv) & y).sum()),
                   "review_other": int(((s >= t_rv) & (s < t_ex) & ~y).sum())}}
print("operating point:", json.dumps(op))

# final probe on everything
X, y = [Xw], [yw]
for v in vids:
    for t, E, _, _, lab in V[v]:
        X.append(E); y.append(np.full(len(E), lab))
X, y = np.concatenate(X), np.concatenate(y)
w = {i: len(y) / (4 * max(1, (y == i).sum())) for i in range(4)}
clf = LogisticRegression(C=2.0, max_iter=3000, class_weight=w).fit(X, y)
np.savez(OUT / "probe.npz", coef=clf.coef_, intercept=clf.intercept_, classes=np.array(C))
(OUT / "operating_point.json").write_text(json.dumps(op, indent=1))
(OUT / "eval.json").write_text(json.dumps({"sweeps": {k: {str(t): r for t, r in v.items()} for k, v in ev.items()},
                                           "tracks": res}, indent=0))
print("saved", OUT)
