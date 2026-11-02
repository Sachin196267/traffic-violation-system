#!/usr/bin/env python3
"""Original-resolution view of crop-sheet cells: zoom_crops.py <sheet> <cell> [<cell> ...]"""
import json, sys, cv2, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent
sheet, cells = sys.argv[1], [int(c) for c in sys.argv[2:]]
video = sheet.rsplit("_", 1)[0][:-1]
names = json.loads((R / "sheets" / f"{video}C_index.json").read_text())[sheet]
units = {r["crop"]: r for r in json.loads((R / "data" / video / "units.json").read_text())}
vp = R.parent / f"helmet_training/robust/vids/{video}.mp4"
cap = cv2.VideoCapture(str(vp if vp.exists() else R.parent / f"datasets/triple_videos/{video}.mp4"))
out = []
for c in cells:
    r = units[names[c - 1]]
    cap.set(1, r["frame"]); ok, f = cap.read()
    x1, y1, x2, y2 = r["box"]; w, h = x2 - x1, y2 - y1
    crop = f[max(0, y1 - h // 6):y2 + h // 10, max(0, x1 - w // 5):x2 + w // 5]
    crop = cv2.resize(crop, None, fx=340 / crop.shape[0], fy=340 / crop.shape[0], interpolation=cv2.INTER_LANCZOS4)
    cv2.putText(crop, str(c), (6, 28), 0, 1.0, (0, 0, 255), 3)
    out.append(cv2.copyMakeBorder(crop, 0, 0, 0, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
rows, cur, W = [], [], 0
for o in out:
    if W + o.shape[1] > 1500 and cur:
        rows.append(cur); cur, W = [], 0
    cur.append(o); W += o.shape[1]
rows.append(cur)
mw = max(sum(o.shape[1] for o in r) for r in rows)
cv2.imwrite("z.jpg",
            np.vstack([cv2.copyMakeBorder(np.hstack(r), 0, 5, 0, mw - sum(o.shape[1] for o in r), cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]))
