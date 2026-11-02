#!/usr/bin/env python3
"""
Camera-stability gate.  Trajectory-based lane inference is only valid while the
camera is fixed, so we track background corners against a reference frame, fit a
similarity transform with RANSAC (moving vehicles become outliers) and split the video into
"static segments".  A pan, zoom or scene cut ends a segment and the lane model
must be rebuilt.
"""
from __future__ import annotations

import cv2
import numpy as np


def _global_motion(ref_g, ref_pts, g):
    """Robust camera motion between ref and g from tracked background corners.
    Returns (shift_px, inlier_frac).  RANSAC ignores moving vehicles."""
    if ref_pts is None or len(ref_pts) < 20:
        return 0.0, 0.0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref_g, g, ref_pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 15:
        return 1e9, 0.0
    M, inl = cv2.estimateAffinePartial2D(ref_pts[ok], nxt[ok], method=cv2.RANSAC,
                                         ransacReprojThreshold=2.0)
    if M is None:
        return 1e9, 0.0
    frac = float(inl.sum()) / len(ref_pts)
    # displacement of the image corners under M = worst-case camera shift
    h, w = g.shape
    c = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32)
    d = c @ M.T - c[:, :2]
    return float(np.abs(d).max()), frac


def static_segments(video: str, max_seconds: float = 120, step_s: float = 0.5,
                    max_shift_px: float = 8.0, min_len_s: float = 3.0, start_s: float = 0.0):
    """Return ([(t_start, t_end), ...], samples) where samples = [(t, shift_px, inlier_frac)].

    Every sample is compared with the first frame of the current segment (not
    just the previous frame), so slow drifts accumulate and get caught too.
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = cap.get(3)
    scale = 640.0 / W
    step = max(1, int(round(step_s * fps)))
    i0 = int(start_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    n = int(min(max_seconds * fps, cap.get(cv2.CAP_PROP_FRAME_COUNT) - i0))
    ref = ref_pts = None
    seg_start, segs, samples = start_s, [], []
    for i in range(n):
        if not cap.grab():
            n = i
            break
        if i % step:
            continue
        _, f = cap.retrieve()
        g = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        t = start_s + i / fps
        if ref is None:
            ref, seg_start = g, t
            ref_pts = cv2.goodFeaturesToTrack(g, 400, 0.01, 8)
            continue
        shift, frac = _global_motion(ref, ref_pts, g)
        samples.append((t, shift / scale, frac))
        # large shift = pan/zoom/shake; low inlier fraction = scene cut
        if shift / scale > max_shift_px or frac < 0.25:
            if t - step_s - seg_start >= min_len_s:
                segs.append((seg_start, t - step_s))
            ref, seg_start = g, t
            ref_pts = cv2.goodFeaturesToTrack(g, 400, 0.01, 8)
    end_t = start_s + n / fps
    if end_t - seg_start >= min_len_s:
        segs.append((seg_start, end_t))
    cap.release()
    return segs, samples


if __name__ == "__main__":
    import sys
    for v in sys.argv[1:]:
        segs, _ = static_segments(v)
        print(v[-40:], [(round(a, 1), round(b, 1)) for a, b in segs])
