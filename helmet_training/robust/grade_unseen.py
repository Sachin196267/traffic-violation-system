#!/usr/bin/env python3
"""
Score run_video.py outputs on the unseen test videos against visual truth labels, for any
operating point (decisions are recomputed from the per-frame scores saved in <video>_riders.json).

truth file (unseen_truth.json): {video: "hnux..."} one letter per rider, in the order of that video's
review sheet in the reference run (h helmet, n no helmet, u can't tell even for a person, x not a rider head).
Riders are matched across runs by (track, slot), which is identical for runs over the same video.

  python grade_unseen.py <run_dir> [<run_dir> ...] --ref video_out/v3_cur
"""
import argparse, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--ref", required=True, help="run whose review-sheet order the truth letters follow")
ap.add_argument("--truth", default=str(ROOT / "unseen_truth.json"))
ap.add_argument("--ops", default="0.8/0.95,0.9/0.98,0.95/0.98", help="thr/nh_thr pairs to report")
args = ap.parse_args()

sys.argv = [sys.argv[0]]
spec = importlib.util.spec_from_file_location("ef", ROOT / "eval_final.py")
ef = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ef)  # decide() identical to run_video.py
C = ef.C
ORDER = {"no_helmet": 0, "not_sure": 1, "helmet": 2}

truth = {k: v.replace(" ", "") for k, v in json.loads(Path(args.truth).read_text()).items() if not k.startswith("_")}
lab = {}
for v, letters in truth.items():
    R = json.loads((Path(args.ref) / f"{v}_riders.json").read_text())["riders"]
    R = sorted(R, key=lambda r: ORDER[r["decision"]])  # review sheet order (stable sort, as in run_video.py)
    assert len(R) == len(letters), (v, len(R), len(letters))
    for r, l in zip(R, letters):
        lab[(v, r["track"], r["slot"])] = l

for run in args.runs:
    for pair in args.ops.split(","):
        thr, nh = map(float, pair.split("/"))
        ef.op.update({"thr": thr, "nh_thr": nh})
        n = defaultdict(int)
        unmatched = 0
        for v in truth:
            for r in json.loads((Path(run) / f"{v}_riders.json").read_text())["riders"]:
                l = lab.get((v, r["track"], r["slot"]))
                if l is None:
                    unmatched += 1
                    continue
                d = ef.decide([(o[0], dict(zip(C, o[1:]))) for o in r["obs"]])
                n[f"{l}->{d}"] += 1
        tot = sum(n.values())
        ns = sum(v for k, v in n.items() if k.endswith("not_sure"))
        called = {k: v for k, v in n.items() if k.split("->")[1] in ("helmet", "no_helmet")}
        wrong_ch = n["h->no_helmet"] + n["x->no_helmet"]
        missed = n["n->helmet"]
        judged = sum(v for k, v in called.items() if k[0] in "hn") + n["x->no_helmet"] + n["x->helmet"]
        print(f"{Path(run).name} thr {thr} nh {nh}: riders {tot} | not sure {ns / max(1, tot):.1%} | "
              f"no_helmet calls {sum(v for k, v in called.items() if k.endswith('no_helmet'))} (wrong {wrong_ch}, unclear {n['u->no_helmet']}) | "
              f"helmet calls {sum(v for k, v in called.items() if k.endswith('->helmet'))} (wrong {missed + n['x->helmet']}, unclear {n['u->helmet']}) | "
              f"wrong among checkable calls {(wrong_ch + missed + n['x->helmet']) / max(1, judged):.1%} | "
              f"no_helmet riders caught {n['n->no_helmet']}/{sum(v for k, v in n.items() if k.startswith('n->'))}"
              + (f" | unmatched {unmatched}" if unmatched else ""))
