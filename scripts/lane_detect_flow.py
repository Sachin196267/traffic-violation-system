#!/usr/bin/env python3
"""
lane_detect_flow.py

Automated lane-boundary detection from vehicle trajectory data, for a FIXED
camera viewing a (possibly curved) road.

WHY THIS APPROACH
------------------
Hough-line-on-background-image detection (your calibrate_lanes.py /
auto_detect_lanes.py path) fails on this scene because:
  1. The road curves, so no single straight line can represent a lane
     boundary across the whole frame.
  2. The strongest straight edges in the background plate belong to the
     pedestrian overpass railings/stairs, not the (faint/worn) lane paint.

This script instead uses the thing you already have plenty of: thousands of
real vehicles driving inside the lanes across the whole video. It clusters
vehicle lateral (x) positions in thin horizontal strips of the image,
propagates cluster identity from strip to strip so lane identity never gets
scrambled, fits a smooth curve through each lane's per-strip center, and
derives boundaries as a variance-weighted split point between adjacent
lanes. Because clustering happens independently per row, road curvature and
perspective are absorbed automatically -- no homography, no manual
calibration, no vanishing-point assumptions.

PIPELINE
--------
1. Load trajectories.csv, auto-detect its column layout.
2. Keep only vehicle-class detections (drop person/rider/bicycle/etc by
   default -- pedestrians on the overpass were almost certainly polluting
   your earlier attempts).
3. Group into tracks, keep tracks that are long enough and "straight enough"
   (path straightness ratio) to be real vehicle motion, not jitter/noise.
4. Take each kept detection's bottom-center point (x, y_bottom) -- the point
   where the vehicle actually touches the road.
5. Bin all points (pooled across the WHOLE video, not one frame) into
   horizontal strips.
6. Cluster each strip's x-values into n_lanes groups with KMeans, warm-started
   from the adjacent strip's centers so lane identity is continuous as you
   move through the image, even through curvature.
7. Fit a smooth polynomial through each lane's (y, center_x) sequence.
8. Derive boundary curves between adjacent lanes (variance-weighted split,
   not a naive midpoint) plus extrapolated outer edges.
9. Save a JSON file of boundary polylines, print a numeric validation report,
   and (optionally) render a preview overlay directly on a video frame.

USAGE
-----
python lane_detect_flow.py \
  --csv "/path/to/trajectories.csv" \
  --video "/path/to/ua_detrac_combined.mp4" \
  --n_lanes 3 \
  --out "/path/to/lane_boundaries_flow.json" \
  --preview "/path/to/lane_flow_preview.jpg"

OUTPUT JSON SCHEMA
------------------
A top-level list of (n_lanes + 1) boundaries, each a list of [x, y] points
sorted by ascending y (top of frame -> bottom), i.e.:

    [
      [[x, y], [x, y], ...],   # boundary 0 (left edge of lane 0)
      [[x, y], [x, y], ...],   # boundary 1 (between lane 0 / lane 1)
      ...
    ]

This mirrors the "click points, sorted by y" convention your own
calibrate_lanes.py / auto_detect_lanes.py already use. If your verify_lanes.py
expects a different wrapper (e.g. {"boundaries": [...]}), the save_boundaries()
function below is the ONLY place that needs to change -- everything else is
independent of the output format.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
except ImportError:
    print("This script needs scikit-learn. Install with:\n"
          "  pip install scikit-learn --break-system-packages")
    sys.exit(1)


DEFAULT_VEHICLE_CLASSES = [
    "car", "truck", "bus", "van", "motorcycle", "vehicle fallback"
]
DEFAULT_EXCLUDE_CLASSES = [
    "person", "rider", "bicycle", "animal", "traffic sign"
]


# --------------------------------------------------------------------------
# 1. CSV loading with auto-detected column names
# --------------------------------------------------------------------------

def detect_columns(df):
    cols = {c.lower().strip(): c for c in df.columns}

    def find(*candidates):
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    frame_col = find("frame", "frame_id", "frame_idx", "frame_number", "frame_num")
    track_col = find("track_id", "id", "tid", "track", "obj_id")
    cls_col = find("class", "cls", "label", "category", "class_name")

    x1 = find("x1", "xmin", "left", "bbox_x1")
    y1 = find("y1", "ymin", "top", "bbox_y1")
    x2 = find("x2", "xmax", "right", "bbox_x2")
    y2 = find("y2", "ymax", "bottom", "bbox_y2")

    cx = find("cx", "center_x", "x_center")
    cy = find("cy", "center_y", "y_center")
    w = find("w", "width", "bbox_w")
    h = find("h", "height", "bbox_h")

    if frame_col is None or track_col is None:
        raise ValueError(
            f"Could not find frame/track_id columns in CSV. "
            f"Columns present: {list(df.columns)}. "
            f"Edit detect_columns() to add your exact column names."
        )

    if x1 and y1 and x2 and y2:
        bbox_mode = "corners"
        bbox_cols = (x1, y1, x2, y2)
    elif cx and cy and w and h:
        bbox_mode = "center"
        bbox_cols = (cx, cy, w, h)
    else:
        raise ValueError(
            f"Could not find bbox columns (need either x1,y1,x2,y2 or "
            f"cx,cy,w,h). Columns present: {list(df.columns)}. "
            f"Edit detect_columns() to add your exact column names."
        )

    return {
        "frame": frame_col,
        "track": track_col,
        "cls": cls_col,
        "bbox_mode": bbox_mode,
        "bbox_cols": bbox_cols,
    }


def load_ground_points(csv_path, vehicle_classes, exclude_classes):
    df = pd.read_csv(csv_path)
    schema = detect_columns(df)
    print(f"Detected CSV schema: {schema}")

    if schema["cls"] is not None:
        cls_series = df[schema["cls"]].astype(str).str.strip().str.lower()
        vc = [c.lower() for c in vehicle_classes]
        ex = [c.lower() for c in exclude_classes]
        before = len(df)
        if vc:
            keep_mask = cls_series.isin(vc)
        else:
            keep_mask = ~cls_series.isin(ex)
        df = df[keep_mask]
        print(f"Class filter: kept {len(df)}/{before} detections "
              f"(classes kept: {sorted(cls_series[keep_mask].unique())})")
    else:
        print("No class column found -- skipping class filtering "
              "(all detections will be used).")

    if schema["bbox_mode"] == "corners":
        x1c, y1c, x2c, y2c = schema["bbox_cols"]
        df["_gx"] = (df[x1c].astype(float) + df[x2c].astype(float)) / 2.0
        df["_gy"] = df[y2c].astype(float)
    else:
        cxc, cyc, wc, hc = schema["bbox_cols"]
        df["_gx"] = df[cxc].astype(float)
        df["_gy"] = df[cyc].astype(float) + df[hc].astype(float) / 2.0

    df["_frame"] = df[schema["frame"]]
    df["_track"] = df[schema["track"]]
    return df[["_frame", "_track", "_gx", "_gy"]].rename(
        columns={"_frame": "frame", "_track": "track", "_gx": "x", "_gy": "y"}
    )


# --------------------------------------------------------------------------
# 2. Track filtering (keep real, smooth vehicle motion; drop jitter/noise)
# --------------------------------------------------------------------------

def filter_tracks(points_df, min_track_len=20, max_turn_deg=35.0,
                   min_smooth_fraction=0.75, min_y_span=25):
    """
    Keep tracks whose motion is LOCALLY smooth (small frame-to-frame heading
    change), rather than globally straight. This is the key fix for curved
    roads: a vehicle following a curving lane can have huge overall
    curvature (low chord/path-length ratio) while still turning only a
    couple of degrees per frame -- that's real driving. A pedestrian on the
    overpass, by contrast, has large, erratic per-step heading changes even
    over a short track. So we filter on per-step turn angle, not on
    straight-line-vs-path-length.
    """
    kept_frames = []
    n_total, n_kept = 0, 0

    for track_id, g in points_df.groupby("track"):
        n_total += 1
        g = g.sort_values("frame")
        if len(g) < min_track_len:
            continue

        y_span = g["y"].max() - g["y"].min()
        if y_span < min_y_span:
            continue

        pts = g[["x", "y"]].to_numpy()
        deltas = np.diff(pts, axis=0)
        mags = np.linalg.norm(deltas, axis=1)
        valid = mags > 1.0  # drop near-static steps (detection jitter, not motion)
        deltas = deltas[valid]
        if len(deltas) < 3:
            continue

        angles = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0]))
        turn = np.diff(angles)
        turn = (turn + 180) % 360 - 180  # wrap to [-180, 180]

        smooth_fraction = np.mean(np.abs(turn) <= max_turn_deg)
        if smooth_fraction < min_smooth_fraction:
            continue

        kept_frames.append(g)
        n_kept += 1

    print(f"Track filtering: kept {n_kept}/{n_total} tracks "
          f"(min_len={min_track_len}, max_turn_deg={max_turn_deg}, "
          f"min_smooth_fraction={min_smooth_fraction}, min_y_span={min_y_span})")

    if not kept_frames:
        raise RuntimeError(
            "No tracks survived filtering -- loosen --min_track_len / "
            "--min_straightness / --min_y_span."
        )
    return pd.concat(kept_frames, ignore_index=True)


# --------------------------------------------------------------------------
# 3. Row-wise (strip-wise) clustering with warm-started continuity
# --------------------------------------------------------------------------

def cluster_strips(points_df, n_lanes, strip_height=15, min_samples=12):
    y_min = int(np.floor(points_df["y"].min()))
    y_max = int(np.ceil(points_df["y"].max()))
    strip_edges = list(range(y_min, y_max + strip_height, strip_height))

    strips = []
    for lo, hi in zip(strip_edges[:-1], strip_edges[1:]):
        mid_y = (lo + hi) / 2.0
        mask = (points_df["y"] >= lo) & (points_df["y"] < hi)
        xs = points_df.loc[mask, "x"].to_numpy()
        strips.append((mid_y, xs))

    strip_counts = [len(xs) for _, xs in strips]
    if max(strip_counts, default=0) < min_samples:
        raise RuntimeError(
            "No strip has enough samples to seed clustering -- lower "
            "--min_samples_per_strip or --strip_height, or check your "
            "class filter isn't too aggressive."
        )
    seed_idx = int(np.argmax(strip_counts))

    centers_by_strip = {}
    stds_by_strip = {}

    def fit_strip(idx, warm_start):
        mid_y, xs = strips[idx]
        if len(xs) < min_samples:
            return None, None
        X = xs.reshape(-1, 1)
        if warm_start is not None:
            km = KMeans(n_clusters=n_lanes, init=warm_start.reshape(-1, 1),
                        n_init=1, random_state=0)
        else:
            km = KMeans(n_clusters=n_lanes, n_init=10, random_state=0)
        km.fit(X)
        centers = km.cluster_centers_.flatten()
        order = np.argsort(centers)
        centers_sorted = centers[order]
        labels = km.labels_
        stds = np.array([
            xs[labels == orig_i].std() if np.sum(labels == orig_i) > 1 else 0.0
            for orig_i in order
        ])
        return centers_sorted, stds

    centers, stds = fit_strip(seed_idx, warm_start=None)
    centers_by_strip[seed_idx] = centers
    stds_by_strip[seed_idx] = stds

    last_good = centers
    for idx in range(seed_idx - 1, -1, -1):
        c, s = fit_strip(idx, warm_start=last_good)
        if c is not None:
            centers_by_strip[idx] = c
            stds_by_strip[idx] = s
            last_good = c

    last_good = centers
    for idx in range(seed_idx + 1, len(strips)):
        c, s = fit_strip(idx, warm_start=last_good)
        if c is not None:
            centers_by_strip[idx] = c
            stds_by_strip[idx] = s
            last_good = c

    lane_center_points = [[] for _ in range(n_lanes)]
    lane_std_points = [[] for _ in range(n_lanes)]
    for idx, (mid_y, _) in enumerate(strips):
        if idx not in centers_by_strip:
            continue
        for lane_i in range(n_lanes):
            lane_center_points[lane_i].append((mid_y, centers_by_strip[idx][lane_i]))
            lane_std_points[lane_i].append((mid_y, stds_by_strip[idx][lane_i]))

    n_strips_used = len(centers_by_strip)
    print(f"Row-wise clustering: used {n_strips_used}/{len(strips)} strips "
          f"(strip_height={strip_height}px, min_samples={min_samples})")

    return lane_center_points, lane_std_points, (y_min, y_max)


# --------------------------------------------------------------------------
# 4. Smooth curve fitting + boundary derivation
# --------------------------------------------------------------------------

def robust_polyfit(y_vals, x_vals, degree=2, n_iter=2, sigma_clip=2.5):
    y_vals = np.asarray(y_vals, dtype=float)
    x_vals = np.asarray(x_vals, dtype=float)
    mask = np.ones(len(y_vals), dtype=bool)
    coeffs = np.polyfit(y_vals, x_vals, degree)
    for _ in range(n_iter):
        pred = np.polyval(coeffs, y_vals[mask])
        resid = x_vals[mask] - pred
        std = resid.std() if resid.std() > 1e-6 else 1e-6
        keep = np.abs(resid) < sigma_clip * std
        if keep.sum() < degree + 2:
            break
        idxs = np.where(mask)[0][keep]
        new_mask = np.zeros(len(y_vals), dtype=bool)
        new_mask[idxs] = True
        mask = new_mask
        coeffs = np.polyfit(y_vals[mask], x_vals[mask], degree)
    return coeffs


def fit_lane_curves(lane_center_points, lane_std_points, degree=2):
    lane_x_fn = []
    lane_std_fn = []
    for pts, spts in zip(lane_center_points, lane_std_points):
        ys = [p[0] for p in pts]
        xs = [p[1] for p in pts]
        coeffs = robust_polyfit(ys, xs, degree=degree)
        lane_x_fn.append(coeffs)

        s_ys = [p[0] for p in spts]
        s_vals = [max(p[1], 1.0) for p in spts]
        std_coeffs = np.polyfit(s_ys, s_vals, min(1, degree))
        lane_std_fn.append(std_coeffs)
    return lane_x_fn, lane_std_fn


def build_boundaries(lane_x_fn, lane_std_fn, y_range, n_points=60,
                      frame_w=None):
    y_min, y_max = y_range
    y_grid = np.linspace(y_min, y_max, n_points)
    n_lanes = len(lane_x_fn)

    lane_centers = np.stack(
        [np.polyval(coeffs, y_grid) for coeffs in lane_x_fn], axis=0
    )  # (n_lanes, n_points)
    lane_stds = np.stack(
        [np.clip(np.polyval(coeffs, y_grid), 1.0, None) for coeffs in lane_std_fn],
        axis=0,
    )

    boundaries = []

    left_gap = lane_centers[1] - lane_centers[0] if n_lanes > 1 else (2 * lane_stds[0])
    left_edge = lane_centers[0] - left_gap / 2.0
    if frame_w is not None:
        left_edge = np.clip(left_edge, 0, frame_w)
    boundaries.append(left_edge)

    for i in range(n_lanes - 1):
        s0, s1 = lane_stds[i], lane_stds[i + 1]
        w = s1 / (s0 + s1 + 1e-6)  # boundary sits closer to the tighter (lower-std) cluster
        split = lane_centers[i] * w + lane_centers[i + 1] * (1 - w)
        boundaries.append(split)

    right_gap = lane_centers[-1] - lane_centers[-2] if n_lanes > 1 else (2 * lane_stds[-1])
    right_edge = lane_centers[-1] + right_gap / 2.0
    if frame_w is not None:
        right_edge = np.clip(right_edge, 0, frame_w)
    boundaries.append(right_edge)

    polylines = []
    for b in boundaries:
        polylines.append([[float(x), float(y)] for x, y in zip(b, y_grid)])
    return polylines


# --------------------------------------------------------------------------
# 5. Numeric validation (don't just trust the picture)
# --------------------------------------------------------------------------

def validate_lane_assignment(points_df, lane_x_fn):
    def assign_lane(x, y):
        centers = [np.polyval(c, y) for c in lane_x_fn]
        return int(np.argmin(np.abs(np.array(centers) - x)))

    consistencies = []
    for track_id, g in points_df.groupby("track"):
        lanes = [assign_lane(row.x, row.y) for row in g.itertuples()]
        if not lanes:
            continue
        vals, counts = np.unique(lanes, return_counts=True)
        consistencies.append(counts.max() / len(lanes))

    if not consistencies:
        print("Validation: no tracks to check.")
        return
    consistencies = np.array(consistencies)
    print(
        f"Validation: mean per-track single-lane consistency = "
        f"{consistencies.mean()*100:.1f}% "
        f"(median {np.median(consistencies)*100:.1f}%, "
        f"{(consistencies >= 0.9).mean()*100:.1f}% of tracks stay in one "
        f"lane >=90% of the time)"
    )
    if consistencies.mean() < 0.75:
        print(
            "  -> Consistency is low. Try a smaller --strip_height, a "
            "higher --min_samples_per_strip, or double-check --n_lanes "
            "matches the real lane count."
        )


# --------------------------------------------------------------------------
# 6. Save + optional preview render
# --------------------------------------------------------------------------

def save_boundaries(boundaries, out_path):
    with open(out_path, "w") as f:
        json.dump(boundaries, f, indent=2)
    print(f"Saved {len(boundaries)} boundaries "
          f"({len(boundaries) - 1} lane(s)) to {out_path}")


def render_preview(video_path, boundaries, frame_index, out_path):
    import cv2

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"Could not read frame {frame_index} from {video_path} "
              f"-- skipping preview.")
        return

    overlay = frame.copy()
    colors = [
        (60, 60, 220), (60, 200, 60), (220, 160, 60),
        (200, 60, 200), (60, 200, 200),
    ]
    n_lanes = len(boundaries) - 1
    for i in range(n_lanes):
        left = np.array(boundaries[i], dtype=np.int32)
        right = np.array(boundaries[i + 1], dtype=np.int32)[::-1]
        poly = np.concatenate([left, right], axis=0)
        cv2.fillPoly(overlay, [poly], colors[i % len(colors)])

    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    for b in boundaries:
        pts = np.array(b, dtype=np.int32)
        cv2.polylines(frame, [pts], False, (255, 255, 255), 2)

    for i in range(n_lanes):
        left = np.array(boundaries[i], dtype=np.float32)
        right = np.array(boundaries[i + 1], dtype=np.float32)
        mid_idx = len(left) // 2
        cx = int((left[mid_idx, 0] + right[mid_idx, 0]) / 2)
        cy = int(left[mid_idx, 1])
        cv2.putText(frame, f"Lane {i}", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, frame)
    print(f"Saved preview to {out_path}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--video", default=None, help="needed only for --preview")
    ap.add_argument("--n_lanes", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strip_height", type=int, default=15)
    ap.add_argument("--min_samples_per_strip", type=int, default=12)
    ap.add_argument("--min_track_len", type=int, default=20)
    ap.add_argument("--max_turn_deg", type=float, default=35.0,
                     help="max per-step heading change (deg) still counted as smooth motion")
    ap.add_argument("--min_smooth_fraction", type=float, default=0.75,
                     help="fraction of a track's steps that must be 'smooth' to keep it")
    ap.add_argument("--min_y_span", type=float, default=25)
    ap.add_argument("--poly_degree", type=int, default=2)
    ap.add_argument("--vehicle_classes", type=str,
                     default=",".join(DEFAULT_VEHICLE_CLASSES),
                     help="comma-separated class names to KEEP. "
                          "Pass an empty string to instead use --exclude_classes.")
    ap.add_argument("--exclude_classes", type=str,
                     default=",".join(DEFAULT_EXCLUDE_CLASSES),
                     help="used only if --vehicle_classes is empty")
    ap.add_argument("--frame_w", type=int, default=None,
                     help="frame width in px, for clipping outer edges "
                          "(auto-read from --video if given)")
    ap.add_argument("--preview", type=str, default=None,
                     help="path to save a preview overlay jpg (needs --video)")
    ap.add_argument("--preview_frame_index", type=int, default=500)
    args = ap.parse_args()

    vehicle_classes = [c.strip() for c in args.vehicle_classes.split(",") if c.strip()]
    exclude_classes = [c.strip() for c in args.exclude_classes.split(",") if c.strip()]

    frame_w = args.frame_w
    if frame_w is None and args.video:
        import cv2
        cap = cv2.VideoCapture(args.video)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        cap.release()

    print("Loading trajectories...")
    points_df = load_ground_points(args.csv, vehicle_classes, exclude_classes)
    print(f"Loaded {len(points_df)} vehicle detections across "
          f"{points_df['track'].nunique()} raw tracks")

    points_df = filter_tracks(
        points_df,
        min_track_len=args.min_track_len,
        max_turn_deg=args.max_turn_deg,
        min_smooth_fraction=args.min_smooth_fraction,
        min_y_span=args.min_y_span,
    )

    lane_center_points, lane_std_points, y_range = cluster_strips(
        points_df,
        n_lanes=args.n_lanes,
        strip_height=args.strip_height,
        min_samples=args.min_samples_per_strip,
    )

    lane_x_fn, lane_std_fn = fit_lane_curves(
        lane_center_points, lane_std_points, degree=args.poly_degree
    )

    boundaries = build_boundaries(lane_x_fn, lane_std_fn, y_range, frame_w=frame_w)

    validate_lane_assignment(points_df, lane_x_fn)
    save_boundaries(boundaries, args.out)

    if args.preview:
        if not args.video:
            print("--preview requires --video, skipping preview render.")
        else:
            render_preview(args.video, boundaries, args.preview_frame_index, args.preview)


if __name__ == "__main__":
    main()
