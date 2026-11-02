#!/usr/bin/env python3
"""
How much video does LaneTrace need?  Re-learn the lanes from the first T
seconds of an observation and compare with
  * the model learnt from the whole observation (stability), and
  * the painted lines (accuracy; only where paint exists).
All errors are in lane widths.  Usage:
    python lane_trace/convergence.py obs1.npz [obs2.npz ...] --out lane_trace/output/convergence
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import score           # noqa: E402
from lane_model import infer_lanes   # noqa: E402

TIMES = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 90, 120, 180, 240, 300, 420, 600]


def main_carriageway(model):
    best = None
    for ri, r in enumerate(model.roads):
        for ci, c in enumerate(r.carriageways):
            if best is None or (c.confidence, c.n_tracks) > (best[2].confidence, best[2].n_tracks):
                best = (ri, r, c, ci)
    return best


def big_carriageways(model, min_tracks=15):
    return [(ri, r, ci, c) for ri, r in enumerate(model.roads) for ci, c in enumerate(r.carriageways)
            if c.n_tracks >= min_tracks]


def line_error(mA, mB, H):
    """For every well-supported carriageway of B: mean image distance (in B's lane
    widths) from each of its boundaries to the nearest boundary of A, and whether
    A has the same number of lanes there.  Returns worst-case over carriageways."""
    linesA = [(r.geometry, b) for r in mA.roads for c in r.carriageways for b in c.boundaries_u]
    centresA = [(r.geometry, c) for r in mA.roads for c in r.carriageways]
    worst_mean, worst_max, same = 0.0, 0.0, True
    for _, rB, _, cB in big_carriageways(mB):
        gB = rB.geometry
        ys = np.linspace(max(gB.y_top, gB.y_h + 0.1 * (H - gB.y_h)), H - 1, 30)
        errs = []
        for b in cB.boundaries_u:
            xb = gB.x_of(b, ys)
            lw = cB.lane_width_u * (ys - gB.y_h)
            errs.append(min(np.mean(np.abs(g.x_of(a, ys) - xb) / lw) for g, a in linesA))
        worst_mean = max(worst_mean, float(np.mean(errs)))
        worst_max = max(worst_max, float(np.max(errs)))
        # lane count of the A-carriageway sitting on the same stretch of road
        yb = 0.92 * H
        xc = gB.x_of(np.mean(cB.centers_u), yb)
        g, cA = min(centresA, key=lambda gc: abs(gc[0].x_of(np.mean(gc[1].centers_u), yb) - xc))
        same &= len(cA.centers_u) == len(cB.centers_u)
    return worst_mean, worst_max, same


def run(path):
    d = np.load(path)
    det, W, H, t0, t1 = d["det"], int(d["W"]), int(d["H"]), float(d["t0"]), float(d["t1"])
    bg = d["bg"]
    dur = t1 - t0
    final = infer_lanes(det, W, H)
    _, _, cF, _ = main_carriageway(final)
    rows = []
    for T in [t for t in TIMES if t < dur] + [dur]:
        dd = det[det[:, 0] < t0 + T]
        row = dict(t=round(T, 1), vehicles=int(len(np.unique(dd[:, 1]))))
        try:
            m = infer_lanes(dd, W, H)
            ri, r, c, ci = main_carriageway(m)
        except Exception as e:
            row["error"] = str(e)[:60]
            rows.append(row)
            continue
        lanes = " + ".join(str(len(cc.centers_u)) for rr in m.roads for cc in rr.carriageways)
        conf = np.mean([cc.confidence for rr in m.roads for cc in rr.carriageways])
        e_mean, e_max, same = line_error(m, final, H)
        row.update(tracks_used=sum(cc.n_tracks for rr in m.roads for cc in rr.carriageways), lanes=lanes,
                   main_lanes=len(c.centers_u), confidence=round(float(conf), 2), same_count_as_final=bool(same),
                   err_vs_final_mean=round(e_mean, 3), err_vs_final_max=round(e_max, 3))
        sc = score(m, bg, ri, ci)
        if len(sc["gt_u"]) >= 2:
            row["paint_divider_err"] = None if sc["divider_err_mean"] is None else round(sc["divider_err_mean"], 3)
            row["paint_edge_err"] = None if sc["edge_err_mean"] is None else round(sc["edge_err_mean"], 3)
            row["paint_matched"] = sc["matched"]
        rows.append(row)
        print(os.path.basename(path), row, flush=True)
    return dict(name=os.path.basename(path), duration_s=round(dur, 1),
                final_lanes=" + ".join(str(len(cc.centers_u)) for rr in final.roads for cc in rr.carriageways),
                final_mode=cF.mode, rows=rows)


def first_stable(rows, tol=0.15):
    """Earliest T from which every later T keeps the final lane count and stays within tol."""
    ok = [("error" not in r) and r.get("same_count_as_final") and r["err_vs_final_max"] <= tol for r in rows]
    for i in range(len(rows)):
        if all(ok[i:]):
            return rows[i]["t"], rows[i]["vehicles"]
    return None, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("obs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = []
    for p in a.obs:
        r = run(p)
        r["stable_from_s"], r["stable_from_vehicles"] = first_stable(r["rows"])
        res.append(r)
        print(r["name"], "stable from", r["stable_from_s"], "s /", r["stable_from_vehicles"], "vehicles")
    with open(os.path.join(a.out, "convergence.json"), "w") as f:
        json.dump(res, f, indent=1)
