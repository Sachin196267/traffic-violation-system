#!/usr/bin/env python3
"""
Group head samples into persons (same track, similar position on the vehicle) and draw labelling sheets.

Each person is labelled once, on its clearest (largest) view; the label is copied to all of that
person's other frames by build_dataset.py. Labels go in helmet_training/robust/labels/<video>.txt
as lines "<pid> <h|n|u|x>":  h helmet, n no helmet, u can't tell, x not a rider head / bad box.

Output: data/<video>/persons.json, sheets/<video>_NN.jpg (48 persons each: best view + a 2nd view).
"""
import json, math, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA, SHEETS = ROOT / "data", ROOT / "sheets"
SHEETS.mkdir(exist_ok=True)
P, COLS, ROWS = 128, 6, 6  # each cell = two patches side by side
MIN_PX = 12  # smaller best views are too blurry to label


def group(rows):
    by_track = {}
    for r in rows:
        by_track.setdefault(r["track"], []).append(r)
    persons = []
    for tid, rs in by_track.items():
        clusters = []
        for r in sorted(rs, key=lambda r: -r["head_px"]):
            best = None
            for c in clusters:
                if abs(c["rel_x"] - r["rel_x"]) < 0.18 and r["frame"] not in c["frames"]:
                    if best is None or abs(c["rel_x"] - r["rel_x"]) < abs(best["rel_x"] - r["rel_x"]):
                        best = c
            if best is None:
                clusters.append({"rel_x": r["rel_x"], "frames": {r["frame"]}, "members": [r["id"]], "rep": r["id"]})
            else:
                best["frames"].add(r["frame"])
                best["members"].append(r["id"])
        for c in clusters:
            persons.append({"track": tid, "members": c["members"], "rep": c["rep"]})
    persons.sort(key=lambda p: min(p["members"]))
    for i, p in enumerate(persons):
        p["pid"] = i
    return persons


def cell(pdir, row, ring=True):
    im = cv2.imread(str(pdir / row["patch"]))
    if ring:
        cv2.rectangle(im, (P // 4 - 2, P // 4 - 2), (3 * P // 4 + 2, 3 * P // 4 + 2), (0, 255, 255), 1)
    return im


MAX_SHEETS = 0
args = sys.argv[1:]
if args and args[0].startswith("--max-sheets="):
    MAX_SHEETS = int(args.pop(0).split("=")[1])
for vd in sorted(DATA.iterdir()) if not args else [DATA / v for v in args]:
    if not (vd / "heads.json").exists():
        continue
    rows = json.loads((vd / "heads.json").read_text())
    byid = {r["id"]: r for r in rows}
    persons = group(rows)
    (vd / "persons.json").write_text(json.dumps(persons))
    per = COLS * ROWS
    todo = [p for p in persons if byid[p["rep"]]["head_px"] >= MIN_PX]
    if MAX_SHEETS and len(todo) > MAX_SHEETS * COLS * ROWS:  # evenly spaced sample across the video
        k = MAX_SHEETS * COLS * ROWS
        todo = [todo[round(i * (len(todo) - 1) / (k - 1))] for i in range(k)]
    for s in range(math.ceil(len(todo) / per)):
        sheet = np.full((ROWS * (P + 18), COLS * (2 * P + 8), 3), 255, np.uint8)
        for k, p in enumerate(todo[s * per:(s + 1) * per]):
            rep = byid[p["rep"]]
            others = [byid[m] for m in p["members"] if m != p["rep"]]
            # second view: a different moment, preferably far in time from the best view
            alt = max(others, key=lambda o: abs(o["frame"] - rep["frame"]) * min(o["head_px"], 30)) if others else None
            a, b = divmod(k, COLS)
            y, x = a * (P + 18) + 18, b * (2 * P + 8)
            sheet[y:y + P, x:x + P] = cell(vd / "patches", rep)
            if alt:
                sheet[y:y + P, x + P:x + 2 * P] = cell(vd / "patches", alt)
            cv2.putText(sheet, f"{p['pid']}  {rep['head_px']:.0f}px n={len(p['members'])}", (x + 2, y - 5), 0, 0.45, (0, 0, 200), 1)
        cv2.imwrite(str(SHEETS / f"{vd.name}_{s:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{vd.name}: {len(rows)} heads -> {len(persons)} persons ({len(todo)} >= {MIN_PX}px on {math.ceil(len(todo) / per)} sheets)")
