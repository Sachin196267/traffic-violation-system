#!/usr/bin/env python3
"""
Crop-level labelling sheets (for scenes where the tracker switches IDs between bikes):
    crop_sheets.py --video indore --min-heads 3 --min-h 90 [--sample N]
36 crops per sheet, numbered.  Record with:  label.py <sheet> "<36 labels>" --crops
Writes sheets/<video>C_<nn>.jpg and sheets/<video>C_index.json (keys = crop file names).
"""
import argparse, json, random
from pathlib import Path
import cv2, numpy as np
R = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--min-heads", type=int, default=3)
ap.add_argument("--min-h", type=int, default=90)
ap.add_argument("--sample", type=int, default=0, help="also add N random crops below the head threshold")
a = ap.parse_args()
U = json.loads((R / "data" / a.video / "units.json").read_text())
done = set()
for lf in (R / "labels").glob("*.crops.txt"):
    done |= {t.split("=")[0] for t in lf.read_text().split()}
pick = [r for r in U if r["n_heads"] >= a.min_heads and r["crop_h"] >= a.min_h and r["crop"] not in done]
rest = [r for r in U if r not in pick and r["crop_h"] >= a.min_h and r["crop"] not in done]
random.Random(0).shuffle(rest)
pick += rest[:a.sample]
pick.sort(key=lambda r: r["t"])
C, index = 200, {}
for s in range(0, len(pick), 36):
    chunk = pick[s:s + 36]
    rows = (len(chunk) + 5) // 6
    sheet = np.full((rows * (C + 22), 6 * (C + 6), 3), 255, np.uint8)
    for i, r in enumerate(chunk):
        r0, c0 = (i // 6) * (C + 22), (i % 6) * (C + 6)
        sheet[r0 + 20:r0 + 20 + C, c0:c0 + C] = cv2.resize(cv2.imread(str(R / "data" / a.video / "crops" / r["crop"])), (C, C))
        cv2.putText(sheet, f"{i + 1:02d}", (c0 + 2, r0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    name = f"{a.video}C_{s // 36:02d}"
    cv2.imwrite(str(R / "sheets" / f"{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    index[name] = [r["crop"] for r in chunk]
(R / "sheets" / f"{a.video}C_index.json").write_text(json.dumps(index))
print(f"{a.video}: {len(pick)} crops -> {len(index)} sheets")
