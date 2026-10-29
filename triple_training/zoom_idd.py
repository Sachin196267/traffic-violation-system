#!/usr/bin/env python3
"""Original-resolution view of IDD units: zoom_idd.py <sheet> <cell> [<cell> ...]"""
import json, sys, glob, cv2, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent
IDD = R.parent / "datasets/kaggle/idd/IDDDetectionsYOLODataset"
sheet, cells = sys.argv[1], [int(c) for c in sys.argv[2:]]
keys = json.loads((R / "sheets" / f"{sheet.rsplit('_', 1)[0]}_index.json").read_text())[sheet]
units = {}
for f in (R / "data").glob("idd_*/units.json"):
    for r in json.loads(f.read_text()):
        units[f"{r['video']}:{r['track']}"] = r
out = []
for c in cells:
    u = units[keys[c - 1]]
    ip = next(p for p in glob.glob(str(IDD / "*/images" / (u["source"] + ".*"))))
    im = cv2.imread(ip)
    x1, y1, x2, y2 = u["box"]; w, h = x2 - x1, y2 - y1
    crop = im[max(0, y1 - h // 6):y2 + h // 10, max(0, x1 - w // 5):x2 + w // 5]
    crop = cv2.resize(crop, None, fx=360 / crop.shape[0], fy=360 / crop.shape[0])
    cv2.putText(crop, str(c), (6, 28), 0, 1.0, (0, 0, 255), 3)
    out.append(cv2.copyMakeBorder(crop, 0, 0, 0, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
rows, cur, W = [], [], 0
for o in out:
    if W + o.shape[1] > 1500 and cur:
        rows.append(cur); cur, W = [], 0
    cur.append(o); W += o.shape[1]
rows.append(cur)
mw = max(sum(o.shape[1] for o in r) for r in rows)
img = np.vstack([cv2.copyMakeBorder(np.hstack(r), 0, 5, 0, mw - sum(o.shape[1] for o in r), cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows])
cv2.imwrite("z.jpg", img)
