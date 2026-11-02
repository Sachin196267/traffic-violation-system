#!/usr/bin/env python3
"""
"Focus" view of a bike crop: everything outside this motorcycle's own column is darkened and
blurred, so riders on neighbouring bikes (the main cause of false triple calls in dense
traffic) fade out while the full context is still visible.  Used identically for training
(make_focus below) and at run time (run_video.py).

The column = motorcycle box widened by 15 % each side; people sitting on the bike are inside
it, a rider on the next bike mostly is not.
"""
import json
from pathlib import Path

import cv2
import numpy as np

SIZE = 256


def letterbox_geom(crop_w, crop_h, size=SIZE):
    s = size / max(crop_w, crop_h)
    nw, nh = max(1, round(crop_w * s)), max(1, round(crop_h * s))
    return s, (size - nw) // 2, (size - nh) // 2


def focus(img, crop_box, moto_box, dim=0.35, widen=0.15):
    """img: letterboxed crop (SIZE x SIZE); crop_box / moto_box in frame pixels."""
    x1, y1, x2, y2 = crop_box
    s, ox, _ = letterbox_geom(x2 - x1, y2 - y1, img.shape[0])
    mw = moto_box[2] - moto_box[0]
    a = int(round((moto_box[0] - widen * mw - x1) * s)) + ox
    b = int(round((moto_box[2] + widen * mw - x1) * s)) + ox
    a, b = max(0, a), min(img.shape[1], b)
    if b - a < 8:
        return img
    out = cv2.GaussianBlur(img, (0, 0), 2.0)
    out = (out.astype(np.float32) * dim + 114 * (1 - dim) * 0.3).astype(np.uint8)
    out[:, a:b] = img[:, a:b]
    return out


def make_focus(root: Path):
    """Write data/<video>/focus/<crop> for every unit that has a motorcycle box."""
    n = 0
    for f in sorted((root / "data").glob("*/units.json")):
        U = json.loads(f.read_text())
        od = f.parent / "focus"
        od.mkdir(exist_ok=True)
        for u in U:
            dst = od / u["crop"]
            if dst.exists() or "moto" not in u:
                continue
            img = cv2.imread(str(f.parent / "crops" / u["crop"]))
            cv2.imwrite(str(dst), focus(img, u["box"], u["moto"]), [cv2.IMWRITE_JPEG_QUALITY, 93])
            n += 1
    print(n, "focus crops written")


if __name__ == "__main__":
    make_focus(Path(__file__).resolve().parent)
