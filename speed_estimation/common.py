"""Shared helpers for the speed tools."""
import json
from pathlib import Path

import cv2
import numpy as np


def background_to_first_frame(run: Path):
    """Homography from run/background.png (median image) to the first analysed frame, which is the coordinate
    frame of tracks.json (track.py registers every frame to it). Identity if there is no background image."""
    tr = json.loads((run / "tracks.json").read_text())
    bg_path = run / "background.png"
    if not bg_path.exists():
        return np.eye(3)
    cap = cv2.VideoCapture(tr["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(tr["t0"] * tr["fps_video"])))
    ok, first = cap.read()
    if not ok:
        return np.eye(3)
    bg = cv2.imread(str(bg_path))
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), None)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 30:
        return np.eye(3)
    H, inl = cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]), np.float32([kb[m.trainIdx].pt for m in good]),
                                cv2.RANSAC, 2.0)
    return H if H is not None else np.eye(3)


def map_points(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, float).reshape(1, -1, 2), H)[0]


def lane_points(cfg, H):
    """Dash end points of a lane-dash config ({vp, lines: {name: {m, dashes: [[y_far, y_near], ...]}}}) given in
    background-image coordinates -> list per line of (distance along line in m, (x, y) in first-frame coords)."""
    vx, vy = cfg["vp"]
    out = {}
    for name, ln in cfg["lines"].items():
        pts = []
        for k, (y_far, y_near) in enumerate(ln["dashes"]):
            pts.append((cfg["cycle_m"] * k, (vx + ln["m"] * (y_far - vy), y_far)))
            if y_near is not None:
                pts.append((cfg["cycle_m"] * k + cfg["dash_m"], (vx + ln["m"] * (y_near - vy), y_near)))
        xy = map_points(H, [p[1] for p in pts])
        out[name] = [(s, tuple(p)) for (s, _), p in zip(pts, xy)]
    return out
