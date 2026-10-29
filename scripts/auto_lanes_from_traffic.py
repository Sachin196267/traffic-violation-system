#!/usr/bin/env python3
"""
auto_lanes_from_traffic.py

FULLY AUTOMATIC lane detection -- no manual clicking anywhere.

Key idea: you already have trajectories.csv from vehicle detection. Vehicles
only ever occupy road surface, never sidewalks/bus stops/medians. So instead
of a human marking the road, we DERIVE the road region automatically from
where vehicles were actually detected, then run lane-marking detection only
inside that region.

This also automatically separates distinct carriageways: if a median
physically divides two roads, vehicle detections on each side form two
spatially disconnected blobs. Connected-component labeling on the vehicle
occupancy mask splits them apart on its own -- no direction/side logic needed.

Pipeline:
1. Load trajectories.csv, keep only vehicle-class detections within a frame
   window around --frame_index (kept local in case the video is a
   concatenation of different camera scenes).
2. Rasterize every vehicle's bounding box onto an occupancy mask, then close
   small gaps with morphology -- this is the automatically-derived "road ROI",
   built entirely from data you already have.
3. Run connectedComponents on that mask to split it into separate physical
   road regions (e.g. near-side vs far-side carriageway, automatically).
4. Drop tiny/noise components (below --min_area_frac of the frame).
5. For EACH remaining road region: extract a median-background plate (to
   cancel out moving vehicles) restricted to that region, run Canny + Hough
   line detection for actual painted lane markings, cluster into boundaries.
6. Save one lane_boundaries JSON per detected road region -- same schema
   lane_utils.py / verify_lanes.py already use.

Usage:
    python auto_lanes_from_traffic.py \
        --csv path/to/trajectories.csv \
        --video path/to/video.mp4 \
        --frame_index 500 \
        --window 250 \
        --n_lanes 3 \
        --out_dir ./lane_output
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np

from auto_detect_lanes import (
    extract_median_background,
    detect_line_segments,
    cluster_segments_to_boundaries,
)

VEHICLE_CLASSES = {1, 3, 4, 6, 13, 14}  # same set used in infer_lanes.py
DEFAULT_BOX_SIZE = 20  # fallback box (px) if width/height columns are missing


def load_vehicle_boxes(csv_path, frame_lo, frame_hi):
    """Return list of (x1, y1, x2, y2) for every vehicle-class detection
    within [frame_lo, frame_hi]."""
    boxes = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        has_wh = reader.fieldnames and "width" in reader.fieldnames and "height" in reader.fieldnames
        for row in reader:
            try:
                cls = int(row["class"])
                frame = int(row["frame"])
            except (KeyError, ValueError):
                continue
            if cls not in VEHICLE_CLASSES:
                continue
            if not (frame_lo <= frame <= frame_hi):
                continue
            cx, cy = float(row["x_center"]), float(row["y_center"])
            if has_wh and row.get("width") and row.get("height"):
                w, h = float(row["width"]), float(row["height"])
            else:
                w, h = DEFAULT_BOX_SIZE, DEFAULT_BOX_SIZE
            boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def build_occupancy_mask(boxes, frame_h, frame_w, close_kernel=25):
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        x2, y2 = min(int(x2), frame_w - 1), min(int(y2), frame_h - 1)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255

    # Close small gaps between nearby vehicle boxes so the road becomes one
    # contiguous region rather than a scatter of disconnected rectangles.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask


def split_into_road_regions(mask, min_area_frac=0.01):
    frame_area = mask.shape[0] * mask.shape[1]
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    regions = []
    for label in range(1, n_labels):  # skip background label 0
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area_frac * frame_area:
            continue
        region_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        regions.append(region_mask)

    # Largest first, so "road_0" is typically the main carriageway
    regions.sort(key=lambda m: cv2.countNonZero(m), reverse=True)
    return regions


def process_one_road(video_path, region_mask, frame_index, window, n_samples, n_lanes):
    h, w = region_mask.shape
    background = extract_median_background(video_path, frame_index, window, n_samples)
    segments = detect_line_segments(background, roi_mask=region_mask)
    if not segments:
        return None
    boundaries = cluster_segments_to_boundaries(segments, h, n_lanes + 1)
    if len(boundaries) < 2:
        return None
    return {
        "frame_width": w,
        "frame_height": h,
        "boundaries": [[[round(x, 1), round(y, 1)] for x, y in b] for b in boundaries],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="trajectories.csv")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame_index", type=int, default=500)
    parser.add_argument("--window", type=int, default=250,
                         help="Frames before/after --frame_index to use for both "
                              "vehicle occupancy and background extraction")
    parser.add_argument("--n_samples", type=int, default=40,
                         help="Frames sampled for median background per road region")
    parser.add_argument("--n_lanes", type=int, default=3,
                         help="Assumed lane count, applied to every detected road region")
    parser.add_argument("--min_area_frac", type=float, default=0.01,
                         help="Drop connected-component road regions smaller than this "
                              "fraction of the frame area (removes noise blobs)")
    parser.add_argument("--out_dir", required=True,
                         help="Directory to write one lane_boundaries JSON per detected road")
    parser.add_argument("--save_occupancy_mask", default=None,
                         help="Optional path to save the raw vehicle-occupancy mask, "
                              "useful for debugging what road area was auto-derived")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    frame_lo = max(args.frame_index - args.window, 0)
    frame_hi = args.frame_index + args.window

    print(f"Loading vehicle detections from {args.csv} in frame range "
          f"[{frame_lo}, {frame_hi}] ...")
    boxes = load_vehicle_boxes(args.csv, frame_lo, frame_hi)
    print(f"Loaded {len(boxes)} vehicle boxes.")
    if len(boxes) < 50:
        print("WARNING: very few vehicle detections in this window -- the "
              "auto-derived road mask may be sparse/unreliable. Consider "
              "increasing --window.")

    occupancy_mask = build_occupancy_mask(boxes, frame_h, frame_w)
    if args.save_occupancy_mask:
        cv2.imwrite(args.save_occupancy_mask, occupancy_mask)
        print(f"Saved occupancy mask to {args.save_occupancy_mask}")

    regions = split_into_road_regions(occupancy_mask, args.min_area_frac)
    print(f"Found {len(regions)} distinct road region(s) after filtering "
          f"components smaller than {args.min_area_frac:.1%} of frame area.")
    if not regions:
        raise SystemExit("No road regions survived filtering. Try lowering "
                          "--min_area_frac or increasing --window for more detections.")

    for i, region_mask in enumerate(regions):
        coverage = cv2.countNonZero(region_mask) / (frame_h * frame_w)
        print(f"\n--- Road region {i} (covers {coverage:.1%} of frame) ---")
        result = process_one_road(args.video, region_mask, args.frame_index,
                                   args.window, args.n_samples, args.n_lanes)
        if result is None:
            print(f"  Could not fit lane boundaries for road region {i} "
                  f"(no usable line segments or too few clusters). Skipped.")
            continue
        out_path = os.path.join(args.out_dir, f"lane_boundaries_road{i}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {len(result['boundaries'])} boundaries "
              f"({len(result['boundaries']) - 1} lane(s)) to {out_path}")

    print("\nDone. Run verify_lanes.py on each lane_boundaries_road*.json "
          "and look at the result before trusting it.")


if __name__ == "__main__":
    main()
