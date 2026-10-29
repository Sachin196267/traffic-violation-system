#!/usr/bin/env python3
"""Contact sheet of e-challan items (crop around each box at its time) for checking by eye.
    python echallan/grade_sheet.py <video> [--decision CHALLAN] -> echallan/runs/<video>/grade_<decision>.jpg"""
import json, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
v = sys.argv[1]; dec = sys.argv[sys.argv.index("--decision") + 1] if "--decision" in sys.argv else "CHALLAN"
run = ROOT / "echallan/runs" / v
J = json.loads((run / "challans.json").read_text())
vid = next(p for p in (ROOT / "datasets").rglob(f"{v}.mp4"))
cap = cv2.VideoCapture(str(vid))
tiles = []
for it in [i for i in J["items"] if i["decision"] == dec]:
    cap.set(cv2.CAP_PROP_POS_MSEC, it["t"] * 1000); ok, fr = cap.read()
    c = np.zeros((200, 200, 3), np.uint8)
    if ok and it.get("box"):
        x1, y1, x2, y2 = map(int, it["box"]); w, h = x2 - x1, y2 - y1; m = max(w, h)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if it["violation"] == "riding without helmet":
            m = max(m * 4, 60); cy += m // 4          # head box -> show the rider too
        else:
            m = int(m * 1.4)
        H, W = fr.shape[:2]
        crop = fr[max(0, cy - m // 2):min(H, cy + m // 2), max(0, cx - m // 2):min(W, cx + m // 2)]
        if crop.size:
            cr = cv2.resize(crop, (200, 200))
            c = cr
    cv2.putText(c, f"{it['id'][-3:]} {it['violation'][:6]} {it['confidence']:.2f}", (2, 194), 0, 0.42, (0, 255, 255), 1)
    tiles.append(c)
while len(tiles) % 8:
    tiles.append(np.zeros((200, 200, 3), np.uint8))
if tiles:
    cv2.imwrite(str(run / f"grade_{dec}.jpg"), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
print(run / f"grade_{dec}.jpg", len([t for t in tiles]))
