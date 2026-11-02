#!/usr/bin/env python3
"""Show original-resolution frames of a track (for checking candidate triples): zoom.py video track [n]"""
import json, sys, cv2, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parent
video, track = sys.argv[1], int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
U = [r for r in json.load(open(R / f"data/{video}/units.json")) if r["track"] == track]
U = sorted(U, key=lambda r: -r["crop_h"])[:n]
U.sort(key=lambda r: r["t"])
vp = R.parent / f"helmet_training/robust/vids/{video}.mp4"
cap = cv2.VideoCapture(str(vp if vp.exists() else R.parent / f"datasets/triple_videos/{video}.mp4"))
out = []
for r in U:
    cap.set(1, r["frame"]); ok, f = cap.read()
    x1, y1, x2, y2 = r["box"]; w, h = x2 - x1, y2 - y1
    c = f[max(0, y1 - h // 4):y2 + h // 8, max(0, x1 - w // 4):x2 + w // 4]
    c = cv2.resize(c, None, fx=380 / c.shape[0], fy=380 / c.shape[0], interpolation=cv2.INTER_LANCZOS4)
    cv2.putText(c, f"{track} t={r['t']}", (5, 20), 0, 0.6, (0, 0, 255), 2)
    out.append(cv2.copyMakeBorder(c, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
cv2.imwrite(sys.argv[4] if len(sys.argv) > 4 else "z.jpg", np.hstack(out))
