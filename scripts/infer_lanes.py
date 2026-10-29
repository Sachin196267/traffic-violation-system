"""
Lane inference from vehicle trajectories.

Pipeline:
1. Load trajectories.csv (frame, track_id, class, x_center, y_center, width, height)
2. Keep only vehicle classes, drop short/noisy tracks
3. For each track, resample its path into fixed y-bins so tracks can be compared
4. Cluster tracks by their x-position pattern (KMeans on the resampled x-values)
5. Fit a polynomial curve per cluster -> this is the inferred lane centerline
6. Plot the result over a sample frame extracted from the source video
"""

import csv
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import argparse
import os

# ---- CONFIG ----
VEHICLE_CLASSES = {1, 3, 4, 6, 13, 14}  # autorickshaw, bus, car, motorcycle, truck, vehicle fallback
MIN_TRACK_LEN = 20          # ignore tracks shorter than this many frames
N_BINS = 20                 # resample each track into this many y-bins for comparison
N_LANES_GUESS = 3           # starting guess for number of lanes (KMeans clusters)
POLY_DEGREE = 2             # degree of polynomial fit per lane


def load_tracks(csv_path):
    tracks = defaultdict(list)  # track_id -> list of (frame, x, y)
    classes = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = int(row["class"])
            if cls not in VEHICLE_CLASSES:
                continue
            tid = int(row["track_id"])
            tracks[tid].append((int(row["frame"]), float(row["x_center"]), float(row["y_center"])))
            classes[tid] = cls
    return tracks, classes


def filter_and_sort(tracks):
    clean = {}
    for tid, pts in tracks.items():
        if len(pts) < MIN_TRACK_LEN:
            continue
        pts.sort(key=lambda p: p[0])  # sort by frame
        clean[tid] = pts
    return clean


def resample_track(pts, y_min, y_max, n_bins=N_BINS):
    """Resample a track's x-position into fixed y-bins (top of frame -> bottom of frame)."""
    ys = np.array([p[2] for p in pts])
    xs = np.array([p[1] for p in pts])

    # Need monotonic-ish y range coverage; sort by y for interpolation
    order = np.argsort(ys)
    ys_sorted = ys[order]
    xs_sorted = xs[order]

    # Remove duplicate y's (interp requires strictly increasing x)
    ys_unique, idx = np.unique(ys_sorted, return_index=True)
    xs_unique = xs_sorted[idx]

    if len(ys_unique) < 2:
        return None

    bin_ys = np.linspace(y_min, y_max, n_bins)
    # Only interpolate within the track's own observed range; mark rest as NaN
    xs_resampled = np.full(n_bins, np.nan)
    valid = (bin_ys >= ys_unique.min()) & (bin_ys <= ys_unique.max())
    xs_resampled[valid] = np.interp(bin_ys[valid], ys_unique, xs_unique)

    return xs_resampled, bin_ys


def build_feature_matrix(clean_tracks, frame_h):
    y_min, y_max = 0, frame_h
    feature_rows = []
    valid_ids = []
    raw_points = {}

    for tid, pts in clean_tracks.items():
        result = resample_track(pts, y_min, y_max)
        if result is None:
            continue
        xs_resampled, bin_ys = result
        # Require solid coverage (at least 65% of bins have data) so partial,
        # perspective-narrow slivers of track don't dominate the clustering.
        if np.sum(~np.isnan(xs_resampled)) < N_BINS * 0.65:
            continue
        feature_rows.append(xs_resampled)
        valid_ids.append(tid)
        raw_points[tid] = pts

    return np.array(feature_rows), valid_ids, bin_ys, raw_points


def cluster_lanes(feature_matrix, n_clusters):
    """
    Fill NaNs per-COLUMN (per y-bin) using the mean x of all tracks that DO have
    data at that y-bin, then cluster.

    Why not row-mean: the road narrows toward the vanishing point, so the same
    lane occupies a much smaller x-range near the top of frame than near the
    bottom. A track only visible near the top and one only visible near the
    bottom can have very different mean x even if they're in the same lane.
    Filling each track's gaps with its OWN mean smears in that perspective
    distortion and misleads KMeans, causing lane clusters (and the curves
    fitted from them) to cross. Column-mean fill keeps every bin's fill value
    anchored to the actual perspective scale at that y position.
    """
    filled = feature_matrix.copy()
    col_means = np.nanmean(filled, axis=0)
    inds = np.where(np.isnan(filled))
    filled[inds] = np.take(col_means, inds[1])

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(filled)
    return labels, km


def robust_polyfit(ys, xs, degree, n_iter=3, sigma_clip=2.5):
    """
    Iteratively-reweighted polynomial fit: fit, drop points whose residual
    exceeds `sigma_clip` standard deviations, refit on the remainder. This
    keeps a handful of mis-clustered points (which are inevitable with
    imperfect clustering) from swinging the whole curve, which is especially
    important near the top of frame where perspective amplifies pixel-x noise.
    """
    keep = np.ones(len(ys), dtype=bool)
    coeffs = np.polyfit(ys, xs, degree)
    for _ in range(n_iter):
        pred = np.polyval(coeffs, ys[keep])
        resid = xs[keep] - pred
        std = resid.std()
        if std == 0 or len(resid) < degree + 2:
            break
        good = np.abs(resid) <= sigma_clip * std
        if good.all():
            break
        # map 'good' (relative to current keep-subset) back onto full-size mask
        idx = np.where(keep)[0]
        keep[idx[~good]] = False
        if keep.sum() < degree + 2:
            break
        coeffs = np.polyfit(ys[keep], xs[keep], degree)
    return coeffs, keep.sum()


def fit_lane_curves(raw_points, valid_ids, labels, n_clusters, frame_h, degree=POLY_DEGREE):
    lane_curves = {}
    for c in range(n_clusters):
        cluster_ids = [tid for tid, lbl in zip(valid_ids, labels) if lbl == c]
        all_x, all_y = [], []
        for tid in cluster_ids:
            for (_, x, y) in raw_points[tid]:
                all_x.append(x)
                all_y.append(y)
        if len(all_x) < 10:
            continue
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        # Fit x as a function of y (since vehicles move mostly along y in typical traffic-cam view),
        # with outlier trimming so mis-clustered points can't dominate the curve.
        coeffs, n_kept = robust_polyfit(all_y, all_x, degree)
        lane_curves[c] = {
            "coeffs": coeffs,
            "y_range": (all_y.min(), all_y.max()),  # observed range, for diagnostics
            "draw_range": (0, frame_h),              # draw across the full frame height
            "n_points": len(all_x),
            "n_points_kept": n_kept,
            "n_tracks": len(cluster_ids),
        }

    lane_curves = _reorder_lanes_left_to_right(lane_curves, frame_h)
    _warn_if_crossing(lane_curves, frame_h)
    return lane_curves


def _reorder_lanes_left_to_right(lane_curves, frame_h):
    """Relabel lanes 0..N-1 by x-position (left to right) at a mid-frame reference line,
    so lane numbering/coloring is stable and meaningful rather than an arbitrary cluster id."""
    ref_y = frame_h * 0.7
    ordered = sorted(lane_curves.items(), key=lambda kv: np.polyval(kv[1]["coeffs"], ref_y))
    return {new_id: info for new_id, (_, info) in enumerate(ordered)}


def _warn_if_crossing(lane_curves, frame_h, n_samples=50):
    """Sanity check: adjacent lanes (after left-to-right ordering) should not cross
    within the drawn range. If they do, the clustering likely mixed two lanes together
    or split one lane in two -- surfaced as a warning rather than silently drawn."""
    ys = np.linspace(0, frame_h, n_samples)
    ids = sorted(lane_curves.keys())
    for a, b in zip(ids, ids[1:]):
        xa = np.polyval(lane_curves[a]["coeffs"], ys)
        xb = np.polyval(lane_curves[b]["coeffs"], ys)
        if np.any(xa > xb):
            print(f"WARNING: Lane {a} and Lane {b} curves cross within the frame. "
                  f"Consider raising --n_lanes, tightening MIN_TRACK_LEN, or checking "
                  f"for mixed traffic directions in the input tracks.")


def draw_lanes_on_frame(frame, lane_curves, out_path):
    h, w = frame.shape[:2]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    for i, (c, info) in enumerate(lane_curves.items()):
        coeffs = info["coeffs"]
        y_lo, y_hi = info["draw_range"]
        ys = np.linspace(y_lo, y_hi, 200)
        xs = np.polyval(coeffs, ys)
        xs = np.clip(xs, 0, w - 1)  # keep extrapolated curve on-canvas
        pts = np.array([xs, ys]).T.astype(np.int32)
        color = colors[i % len(colors)]
        for j in range(len(pts) - 1):
            cv2.line(frame, tuple(pts[j]), tuple(pts[j + 1]), color, 3)
        label_pos = tuple(pts[len(pts) // 2])
        cv2.putText(frame, f"Lane {c} (n={info['n_tracks']})", label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(out_path, frame)
    print(f"Saved lane visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to trajectories.csv")
    parser.add_argument("--video", required=True, help="Path to source video (to grab a sample frame)")
    parser.add_argument("--frame_index", type=int, default=500, help="Which frame to draw lanes on")
    parser.add_argument("--n_lanes", type=int, default=N_LANES_GUESS, help="Number of lanes to assume")
    parser.add_argument("--out", default="lane_overlay.jpg", help="Output image path")
    args = parser.parse_args()

    print("Loading tracks...")
    tracks, classes = load_tracks(args.csv)
    print(f"Loaded {len(tracks)} raw tracks")

    clean = filter_and_sort(tracks)
    print(f"{len(clean)} tracks remain after filtering (min length {MIN_TRACK_LEN})")

    cap = cv2.VideoCapture(args.video)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Video resolution: {frame_w}x{frame_h}")

    feature_matrix, valid_ids, bin_ys, raw_points = build_feature_matrix(clean, frame_h)
    print(f"{len(valid_ids)} tracks have enough vertical coverage to use for lane fitting")

    if len(valid_ids) < args.n_lanes * 3:
        print("WARNING: very few usable tracks — lane estimate will be rough. "
              "Consider using a longer/less-cut video or lowering MIN_TRACK_LEN.")

    labels, km = cluster_lanes(feature_matrix, args.n_lanes)

    lane_curves = fit_lane_curves(raw_points, valid_ids, labels, args.n_lanes, frame_h)
    print(f"Fitted {len(lane_curves)} lane curves")
    MIN_TRACKS_PER_LANE = 12
    for c, info in lane_curves.items():
        flag = ""
        if info["n_tracks"] < MIN_TRACKS_PER_LANE:
            flag = f"  <-- WEAK: fewer than {MIN_TRACKS_PER_LANE} tracks, curve may be unreliable"
        print(f"  Lane {c}: {info['n_tracks']} tracks, {info['n_points']} points "
              f"({info['n_points_kept']} kept after outlier trim), "
              f"observed y-range {info['y_range'][0]:.0f}-{info['y_range'][1]:.0f}{flag}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame {args.frame_index}, falling back to frame 0")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    draw_lanes_on_frame(frame, lane_curves, args.out)
    cap.release()


if __name__ == "__main__":
    main()
