#!/usr/bin/env python3
"""
auto_detect_lanes.py

Fully automatic lane boundary detection from a FIXED camera, based on the
actual painted lane markings in the video -- not on vehicle trajectories.

REQUIRES a road ROI first (run select_roi.py once): without it, Hough line
detection has no concept of "road" and will happily pick up curbs, railings,
sidewalk edges, and opposite-direction lanes as if they were lane markings.

Pipeline:
1. Sample N frames from a local window around --frame_index (NOT the whole
   video, in case it's a concatenation of different camera scenes) and take
   the pixel-wise MEDIAN across them. Moving vehicles cancel out; the static
   road + lane markings remain -- this gives a clean background plate.
2. Mask that background to ONLY the road ROI polygon (from select_roi.py).
3. Edge-detect (Canny) inside the mask and run a Hough Line Transform to find
   straight line segments -- this is what actually catches painted lines.
4. Discard near-horizontal segments (crosswalks, curbs, shadows across the
   road) -- lane dividers run roughly along the direction of travel.
5. Cluster the remaining segments by where they'd cross the bottom of the
   frame (KMeans on x-intercept), fit a curve through each cluster.
6. Save boundaries in the same JSON schema calibrate_lanes.py produces, so
   verify_lanes.py and lane_utils.py work unchanged.

IMPORTANT: this is a heuristic detector, not ground truth. It depends on lane
markings actually being visible, painted, and not too worn/occluded. Always
run verify_lanes.py afterward and check the result.

Usage:
    python select_roi.py --video path/to/video.mp4 --frame_index 500 --out road_roi.json
    python auto_detect_lanes.py \
        --video path/to/video.mp4 \
        --frame_index 500 \
        --window 250 \
        --n_samples 40 \
        --n_lanes 3 \
        --roi road_roi.json \
        --out lane_boundaries_auto.json
"""

import argparse
import json

import cv2
import numpy as np
from sklearn.cluster import KMeans


def sample_frame_indices(total_frames, center, window, n_samples):
    lo = max(center - window, 0)
    hi = min(center + window, total_frames - 1)
    if hi <= lo:
        return [center]
    return sorted(set(np.linspace(lo, hi, n_samples).astype(int).tolist()))


def extract_median_background(video_path, frame_index, window, n_samples):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = sample_frame_indices(total_frames, frame_index, window, n_samples)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if len(frames) < 5:
        raise SystemExit(f"Only read {len(frames)} usable frames -- need more "
                          f"for a clean median background. Check --window/--n_samples.")

    stack = np.stack(frames, axis=0)
    background = np.median(stack, axis=0).astype(np.uint8)
    print(f"Built background from {len(frames)} frames "
          f"(indices {indices[0]}..{indices[-1]}).")
    return background


def load_roi_mask(roi_json_path, frame_h, frame_w):
    with open(roi_json_path) as f:
        data = json.load(f)
    polygon = np.array(data["polygon"], dtype=np.int32)
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def detect_line_segments(background_bgr, roi_mask=None, min_angle_deg=25):
    gray = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    if roi_mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

    raw_segments = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=35,
        minLineLength=35, maxLineGap=25,
    )
    if raw_segments is None:
        return []

    # OpenCV's HoughLinesP has returned (N, 1, 4) historically and (N, 4) in
    # some newer builds (e.g. opencv-python 5.x) -- normalize to (N, 4).
    raw_segments = np.asarray(raw_segments).reshape(-1, 4)

    kept = []
    for seg in raw_segments:
        x1, y1, x2, y2 = seg
        dx, dy = (x2 - x1), (y2 - y1)
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        angle = min(angle, 180 - angle)  # fold into [0, 90]
        # Keep segments that are steep enough to plausibly be a lane divider
        # receding toward a vanishing point (i.e. NOT near-horizontal, which
        # is usually a crosswalk stripe, curb, or shadow across the road).
        if angle >= min_angle_deg:
            kept.append((x1, y1, x2, y2))
    print(f"Detected {len(raw_segments)} raw line segments, "
          f"{len(kept)} kept after angle filtering.")
    return kept


def x_intercept_at(seg, ref_y):
    x1, y1, x2, y2 = seg
    if y1 == y2:
        return (x1 + x2) / 2.0
    t = (ref_y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def cluster_segments_to_boundaries(segments, frame_h, n_boundaries):
    ref_y = frame_h  # bottom of frame
    intercepts = np.array([[x_intercept_at(s, ref_y)] for s in segments])

    km = KMeans(n_clusters=n_boundaries, n_init=10, random_state=42)
    labels = km.fit_predict(intercepts)

    # Order clusters left-to-right by intercept so boundary indices are meaningful
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])

    boundaries = []
    for b in range(n_boundaries):
        cluster_segs = [seg for seg, lbl in zip(segments, labels) if lbl == b]
        if not cluster_segs:
            print(f"WARNING: boundary {b} got no segments -- try adjusting "
                  f"--n_lanes, or the markings for this boundary may be too "
                  f"faint/occluded for auto-detection.")
            continue
        xs, ys = [], []
        for x1, y1, x2, y2 in cluster_segs:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        xs, ys = np.array(xs), np.array(ys)
        coeffs = np.polyfit(ys, xs, deg=2 if len(xs) >= 8 else 1)
        y_lo, y_hi = ys.min(), ys.max()
        sample_ys = np.linspace(y_lo, y_hi, 20)
        sample_xs = np.polyval(coeffs, sample_ys)
        boundaries.append(list(zip(sample_xs.tolist(), sample_ys.tolist())))
    return boundaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame_index", type=int, default=500,
                         help="Center frame of the scene to analyze")
    parser.add_argument("--window", type=int, default=250,
                         help="Frames before/after --frame_index to sample from "
                              "(keep this within ONE continuous camera scene)")
    parser.add_argument("--n_samples", type=int, default=40,
                         help="How many frames to sample for the median background")
    parser.add_argument("--n_lanes", type=int, default=3)
    parser.add_argument("--roi", required=True,
                         help="Path to road_roi.json from select_roi.py -- restricts "
                              "detection to the actual carriageway, excluding "
                              "sidewalks/opposite-direction lanes/medians.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--save_background", default=None,
                         help="Optional path to also save the median background "
                              "image, useful for debugging if detection looks off")
    parser.add_argument("--save_masked_background", default=None,
                         help="Optional path to save the background with the ROI "
                              "mask applied, so you can confirm the mask covers "
                              "the right road area before trusting the result")
    args = parser.parse_args()

    background = extract_median_background(args.video, args.frame_index,
                                             args.window, args.n_samples)
    if args.save_background:
        cv2.imwrite(args.save_background, background)
        print(f"Saved background plate to {args.save_background}")

    h, w = background.shape[:2]
    roi_mask = load_roi_mask(args.roi, h, w)

    if args.save_masked_background:
        masked_preview = cv2.bitwise_and(background, background, mask=roi_mask)
        cv2.imwrite(args.save_masked_background, masked_preview)
        print(f"Saved masked background (everything outside ROI blacked out) "
              f"to {args.save_masked_background}")

    segments = detect_line_segments(background, roi_mask=roi_mask)
    if not segments:
        raise SystemExit("No line segments detected at all. The lane markings may be "
                          "too faint/worn, or thresholds need loosening in detect_line_segments().")

    n_boundaries = args.n_lanes + 1
    boundaries = cluster_segments_to_boundaries(segments, h, n_boundaries)

    if len(boundaries) < 2:
        raise SystemExit("Fewer than 2 boundaries survived clustering -- "
                          "cannot form a lane. See warnings above.")

    data = {
        "frame_width": w,
        "frame_height": h,
        "boundaries": [[[round(x, 1), round(y, 1)] for x, y in b] for b in boundaries],
    }
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(boundaries)} auto-detected boundaries "
          f"({len(boundaries) - 1} lane(s)) to {args.out}")
    print("Now run verify_lanes.py on this file and LOOK at the result -- "
          "auto-detection is a starting point, not guaranteed ground truth.")


if __name__ == "__main__":
    main()
