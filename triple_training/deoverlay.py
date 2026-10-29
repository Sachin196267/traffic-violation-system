#!/usr/bin/env python3
"""
Remove police-video graphics (bright red circles / arrows, red / green / blue circles and plate boxes, yellow plate text) from
crops of enforcement clips, so the classifier cannot learn "overlay = violation".  Applied to
EVERY crop of those videos (triples and non-triples alike).  Saturated overlay colours are found
in HSV, dilated a little and inpainted.
    python triple_training/deoverlay.py            # all videos named tpt-*, hydpol-*
Writes data/<video>/crops_clean/<crop>; build_dataset.py uses them when present.
"""
import glob
from pathlib import Path

import cv2
import numpy as np

R = Path(__file__).resolve().parent


def clean(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h <= 7) | (h >= 170)) & (s >= 120) & (v >= 110)
    yellow = (h >= 20) & (h <= 38) & (s >= 110) & (v >= 150)
    green = (h >= 45) & (h <= 85) & (s >= 140) & (v >= 140)
    blue = (h >= 105) & (h <= 130) & (s >= 150) & (v >= 120)
    m = (red | yellow | green | blue).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # ignore single pixels
    if m.sum() == 0:
        return img
    m = cv2.dilate(m, np.ones((7, 7), np.uint8))
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA)


if __name__ == "__main__":
    n = 0
    for d in sorted(glob.glob(str(R / "data/tpt-*")) + glob.glob(str(R / "data/hydpol-*"))):
        d = Path(d)
        out = d / "crops_clean"
        out.mkdir(exist_ok=True)
        for f in sorted((d / "crops").glob("*.jpg")):
            dst = out / f.name
            if not dst.exists():
                cv2.imwrite(str(dst), clean(cv2.imread(str(f))), [cv2.IMWRITE_JPEG_QUALITY, 93])
                n += 1
    print(n, "crops cleaned")
