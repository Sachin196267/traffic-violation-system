#!/usr/bin/env python3
"""
verify_lanes.py

Draws the manually-calibrated lane boundaries (from calibrate_lanes.py) onto
a video frame so you can visually confirm they actually line up with the
real lane markings before trusting them in the violation-detection pipeline.

Usage:
    python verify_lanes.py \
        --video path/to/video.mp4 \
        --boundaries lane_boundaries.json \
        --frame_index 500 \
        --out lane_verify.jpg
"""

import argparse

import cv2
import numpy as np

from lane_utils import LaneMap

PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--boundaries", required=True, help="lane_boundaries.json from calibrate_lanes.py")
    parser.add_argument("--frame_index", type=int, default=500)
    parser.add_argument("--out", default="lane_verify.jpg")
    parser.add_argument("--fill_alpha", type=float, default=0.25,
                         help="Opacity of the per-lane fill overlay (0 disables fill).")
    args = parser.parse_args()

    lm = LaneMap(args.boundaries)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit(f"Could not read frame {args.frame_index}")

    overlay = frame.copy()

    for lane_idx in range(lm.n_lanes):
        color = PALETTE[lane_idx % len(PALETTE)]
        poly = lm.lane_polygon(lane_idx)
        if args.fill_alpha > 0:
            cv2.fillPoly(overlay, [poly], color)
        cx, cy = poly.mean(axis=0).astype(int)
        cv2.putText(frame, f"Lane {lane_idx}", (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if args.fill_alpha > 0:
        frame = cv2.addWeighted(overlay, args.fill_alpha, frame, 1 - args.fill_alpha, 0)

    for i, (ys, xs) in enumerate(lm.boundaries):
        pts = np.array([[np.interp(y, ys, xs), y] for y in
                         np.linspace(ys.min(), ys.max(), 100)], dtype=np.int32)
        for j in range(len(pts) - 1):
            cv2.line(frame, tuple(pts[j]), tuple(pts[j + 1]), (255, 255, 255), 2)

    cv2.imwrite(args.out, frame)
    print(f"Saved verification image to {args.out}")


if __name__ == "__main__":
    main()
