"""
robust_lane_fit.py

Fixes the failure mode visible in lane_flow_preview.jpg: "Lane 0" ballooning
out across the sidewalk / bus shelter / pedestrian area instead of hugging
the actual road.

ROOT CAUSES (inferred from the terminal log + preview image):

  1. Boundary curves were evaluated across the FULL frame height even in
     rows where that lane had little or no supporting evidence. Polynomial /
     spline fits are unconstrained outside their support range and blow up
     sideways -- exactly the bulge shape in the screenshot.
  2. Nothing clipped the fitted curve to the road ROI, so a boundary was
     free to sweep into non-road pixels.
  3. Lanes were fit independently rather than jointly, so nothing enforced
     that boundaries stay ordered left-to-right and never cross -- which is
     consistent with the low track-to-lane consistency you measured
     (75.7% mean, only 29.3% of tracks staying in one lane >=90% of the time).

FIX: an EM "mixture of curves" fit.

  - Each lane is a low-degree curve x = f(y) fit with a ROBUST regressor
    (RANSAC), not a plain polynomial least-squares fit, so outlier points
    (pedestrians/bicycles that slipped through class filtering, jittery
    boxes, etc.) don't drag the curve.
  - Each lane curve carries an explicit SUPPORT RANGE [y_min, y_max] = the
    actual range of its assigned evidence. Outside that range the curve is
    extended with a DAMPED LINEAR extrapolation, capped in pixels -- never
    raw polynomial curvature. This alone kills the balloon shape.
  - After every M-step, lanes are checked for ordering/crossing at a dense
    set of sampled rows. If two lanes cross, that k is rejected and the
    fit is retried at k-1 lanes (lane count is evidence-driven, not fixed).
  - Every boundary point is clipped to the road ROI polygon if one is
    supplied, otherwise to the frame bounds.

This is a building block, not the full 40-step pipeline described in your
spec -- it targets the specific defect in the current preview. It's meant
to slot in as the "lane hypothesis generation + boundary generation +
confidence" stage, replacing whatever curve-fit code produced
lane_flow_preview.jpg. It expects the same CSV schema your pipeline
already detected:

    frame, track_id, class, x_center, y_center, width, height  (bbox_mode=center)

Usage:
    python robust_lane_fit.py \
        --csv datasets/trajectories.csv \
        --video datasets/ua_detrac_combined.mp4 \
        --vehicle_classes 1,3,4,5,6,13,14 \
        --roi road_roi.json \
        --out_dir lane_output/robust \
        --max_lanes 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import RANSACRegressor, LinearRegression
except ImportError:
    print("Missing dependency: pip install scikit-learn --break-system-packages")
    sys.exit(1)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class LaneFitConfig:
    vehicle_classes: list[str]
    min_track_len: int = 20
    min_y_span: int = 25
    max_turn_deg: float = 35.0
    degree: int = 1                    # 1 = straight, 2 = mild curve
    max_lanes: int = 6
    min_lanes: int = 1
    em_iters: int = 12
    extrapolation_cap_px: int = 60     # hard cap on how far a curve may be
                                        # pushed beyond its own evidence
    min_points_per_lane: int = 30      # below this, the lane is dropped
    ransac_residual_px: float = 12.0


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_tracks(csv_path: str, vehicle_classes: list[str]) -> dict[int, np.ndarray]:
    """Return {track_id: Nx2 array of (y_bottom, x_bottom)} sorted by y,
    restricted to the given vehicle classes only."""
    df = pd.read_csv(csv_path)
    df["class"] = df["class"].astype(str)
    df = df[df["class"].isin(vehicle_classes)]

    # bottom-center = road contact point, per the spec
    df["x_bottom"] = df["x_center"]
    df["y_bottom"] = df["y_center"] + df["height"] / 2.0

    tracks: dict[int, np.ndarray] = {}
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        pts = g[["y_bottom", "x_bottom"]].to_numpy(dtype=float)
        tracks[int(tid)] = pts
    return tracks


def filter_tracks(tracks: dict[int, np.ndarray], cfg: LaneFitConfig) -> dict[int, np.ndarray]:
    kept = {}
    for tid, pts in tracks.items():
        if len(pts) < cfg.min_track_len:
            continue
        y_span = pts[:, 0].max() - pts[:, 0].min()
        if y_span < cfg.min_y_span:
            continue
        # straightness check: reject tracks that zigzag (likely bad
        # associations / ID switches), using turn angle between consecutive
        # displacement vectors
        if len(pts) >= 3:
            deltas = np.diff(pts, axis=0)
            angles = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0]))
            turn = np.abs(np.diff(angles))
            turn = np.minimum(turn, 360 - turn)
            if np.median(turn) > cfg.max_turn_deg:
                continue
        kept[tid] = pts
    if not kept:
        raise RuntimeError(
            "No tracks survived filtering -- loosen min_track_len / "
            "min_y_span / max_turn_deg, or check that vehicle_classes "
            "actually matches your CSV's class values."
        )
    return kept


# --------------------------------------------------------------------------
# Lane curve: robust fit + support range + damped extrapolation
# --------------------------------------------------------------------------

@dataclass
class LaneCurve:
    model: object
    degree: int
    y_min: float
    y_max: float
    n_points: int
    residual_px: float
    cap_px: int

    def eval(self, y: np.ndarray) -> np.ndarray:
        y = np.atleast_1d(np.asarray(y, dtype=float))
        X = self._feat(y)
        x = self.model.predict(X)

        # damped extrapolation outside the support range: beyond y_min/y_max
        # we do NOT let the polynomial keep curving. We linearly continue
        # from the curve's own edge slope, then clamp total displacement.
        below = y < self.y_min
        above = y > self.y_max
        if np.any(below):
            x_edge = self.model.predict(self._feat(np.array([self.y_min])))[0]
            slope = self._edge_slope(self.y_min, forward=True)
            dy = y[below] - self.y_min
            delta = np.clip(slope * dy, -self.cap_px, self.cap_px)
            x[below] = x_edge + delta
        if np.any(above):
            x_edge = self.model.predict(self._feat(np.array([self.y_max])))[0]
            slope = self._edge_slope(self.y_max, forward=False)
            dy = y[above] - self.y_max
            delta = np.clip(slope * dy, -self.cap_px, self.cap_px)
            x[above] = x_edge + delta
        return x

    def _feat(self, y: np.ndarray) -> np.ndarray:
        if self.degree == 1:
            return y.reshape(-1, 1)
        return np.stack([y, y ** 2], axis=1)

    def _edge_slope(self, y_at: float, forward: bool) -> float:
        h = 2.0
        y0 = y_at - h if forward else y_at - h
        y1 = y_at
        x0 = self.model.predict(self._feat(np.array([max(y0, self.y_min)])))[0]
        x1 = self.model.predict(self._feat(np.array([min(y1, self.y_max)])))[0]
        if y1 == y0:
            return 0.0
        return (x1 - x0) / (y1 - y0)


def fit_curve(points: np.ndarray, degree: int, residual_px: float, cap_px: int) -> Optional[LaneCurve]:
    """points: Nx2 (y, x). Fit x = f(y) robustly."""
    if len(points) < 8:
        return None
    y = points[:, 0]
    x = points[:, 1]
    X = y.reshape(-1, 1) if degree == 1 else np.stack([y, y ** 2], axis=1)

    try:
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=residual_px,
            min_samples=max(8, int(0.5 * len(points))),
            max_trials=200,
        )
        ransac.fit(X, x)
    except ValueError:
        return None

    inlier_mask = ransac.inlier_mask_
    resid = np.abs(x[inlier_mask] - ransac.predict(X[inlier_mask]))
    return LaneCurve(
        model=ransac,
        degree=degree,
        y_min=float(y.min()),
        y_max=float(y.max()),
        n_points=int(inlier_mask.sum()),
        residual_px=float(np.median(resid)) if len(resid) else float("inf"),
        cap_px=cap_px,
    )


# --------------------------------------------------------------------------
# EM mixture-of-curves lane fitting
# --------------------------------------------------------------------------

def _track_mean_x(pts: np.ndarray) -> float:
    return float(np.median(pts[:, 1]))


def em_fit_lanes(tracks: dict[int, np.ndarray], k: int, cfg: LaneFitConfig) -> Optional[list[LaneCurve]]:
    ids = list(tracks.keys())
    if len(ids) < k:
        return None

    # init: sort tracks by median x, split into k evenly-sized groups
    order = sorted(ids, key=lambda i: _track_mean_x(tracks[i]))
    groups = np.array_split(order, k)
    assignment = {tid: g_idx for g_idx, g in enumerate(groups) for tid in g}

    curves: list[Optional[LaneCurve]] = [None] * k

    for _ in range(cfg.em_iters):
        # M-step: refit each lane from its assigned tracks' points
        for lane_idx in range(k):
            pts = np.concatenate(
                [tracks[tid] for tid, a in assignment.items() if a == lane_idx],
                axis=0,
            ) if any(a == lane_idx for a in assignment.values()) else np.empty((0, 2))
            curves[lane_idx] = fit_curve(pts, cfg.degree, cfg.ransac_residual_px, cfg.extrapolation_cap_px)

        if any(c is None for c in curves):
            return None  # this k is not supported by the evidence

        # E-step: reassign whole tracks to whichever curve best matches
        # (assignment is per-track, not per-point, so a single track can't
        # be split across two "lanes" mid-track)
        new_assignment = {}
        for tid, pts in tracks.items():
            y = pts[:, 0]
            best_idx, best_err = None, None
            for lane_idx, curve in enumerate(curves):
                pred = curve.eval(y)
                err = np.mean(np.abs(pred - pts[:, 1]))
                if best_err is None or err < best_err:
                    best_err, best_idx = err, lane_idx
            new_assignment[tid] = best_idx

        if new_assignment == assignment:
            break
        assignment = new_assignment

    # order lanes left-to-right at the bottom row, then verify non-crossing
    ref_y = max(c.y_max for c in curves)
    order_idx = sorted(range(k), key=lambda i: curves[i].eval(np.array([ref_y]))[0])
    curves = [curves[i] for i in order_idx]

    if _has_crossing(curves, ref_y):
        return None  # caller should retry with k-1

    # drop lanes with too little evidence
    curves = [c for c in curves if c.n_points >= cfg.min_points_per_lane]
    if len(curves) < 1:
        return None
    return curves


def _has_crossing(curves: list[LaneCurve], ref_y: float) -> bool:
    y_lo = min(c.y_min for c in curves)
    y_hi = max(c.y_max for c in curves)
    sample_y = np.linspace(y_lo, y_hi, 40)
    xs = np.stack([c.eval(sample_y) for c in curves], axis=1)  # (40, k)
    diffs = np.diff(xs, axis=1)
    return bool(np.any(diffs <= 0))


def estimate_lane_count(tracks: dict[int, np.ndarray], cfg: LaneFitConfig) -> tuple[Optional[list[LaneCurve]], int]:
    """Try k from max_lanes down to min_lanes, keep the largest k that
    produces a valid, non-crossing, sufficiently-supported fit."""
    for k in range(cfg.max_lanes, cfg.min_lanes - 1, -1):
        curves = em_fit_lanes(tracks, k, cfg)
        if curves is not None and len(curves) == k:
            return curves, k
    return None, 0


# --------------------------------------------------------------------------
# Drivable-area mask (data-driven ROI)
#
# Instead of trusting a single hand-drawn polygon (which is a pain to get
# right around an irregular bus bay / curb notch), we build the road mask
# from the evidence we already have: everywhere a vehicle's footprint has
# ever been recorded. Cars don't drive across the sidewalk, the bus
# shelter, or the fenced strip on the right -- so the closed union of all
# vehicle footprints IS the drivable surface, notches and all, with no
# manual polygon needed. An optional hand-drawn --roi polygon can still be
# supplied and is intersected in as an extra hard limit (useful near the
# frame edges / far distance where detections are sparse).
# --------------------------------------------------------------------------

def load_roi(roi_path: Optional[str], frame_shape: tuple[int, int]) -> Optional[np.ndarray]:
    h, w = frame_shape
    if roi_path and os.path.exists(roi_path):
        with open(roi_path) as f:
            data = json.load(f)
        pts = np.array(data["polygon"], dtype=np.int32)
        return pts
    return None  # None == "no extra hand-drawn constraint", not "whole frame"


def build_drivable_mask(
    csv_path: str,
    vehicle_classes: list[str],
    frame_shape: tuple[int, int],
    roi_poly: Optional[np.ndarray] = None,
    close_px: int = 41,
    dilate_px: int = 18,
) -> np.ndarray:
    """Rasterize every vehicle footprint (from the RAW, unfiltered detections
    -- not just the tracks that survived straightness/length filtering, so
    we get maximum coverage) into a binary road mask, then clean it up.

    Steps:
      1. For every detection, draw its bbox footprint (not just a point) so
         a lane's full width gets covered, not just a 1px trajectory line.
      2. Morphological CLOSE to bridge small gaps between passing vehicles
         (e.g. a lane that's briefly empty in the frame).
      3. Morphological DILATE a little so the mask is generous rather than
         eating into real lane edges.
      4. Keep only the LARGEST connected component -- this is what silently
         drops stray/misclassified detections that landed on the sidewalk
         (a single bad box shouldn't open a hole to the pedestrian area).
      5. Intersect with the hand-drawn --roi polygon if one was supplied.
    """
    h, w = frame_shape
    df = pd.read_csv(csv_path)
    df["class"] = df["class"].astype(str)
    df = df[df["class"].isin(vehicle_classes)]

    mask = np.zeros((h, w), dtype=np.uint8)
    xc = df["x_center"].to_numpy(dtype=float)
    yc = df["y_center"].to_numpy(dtype=float)
    bw = df["width"].to_numpy(dtype=float)
    bh = df["height"].to_numpy(dtype=float)
    for i in range(len(df)):
        x0 = int(round(xc[i] - bw[i] / 2))
        x1 = int(round(xc[i] + bw[i] / 2))
        y0 = int(round(yc[i] - bh[i] / 2))
        y1 = int(round(yc[i] + bh[i] / 2))
        cv2.rectangle(mask, (max(0, x0), max(0, y0)),
                      (min(w - 1, x1), min(h - 1, y1)), 255, -1)

    if close_px > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    if dilate_px > 0:
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k_dilate)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    if roi_poly is not None:
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [roi_poly], 255)
        mask = cv2.bitwise_and(mask, roi_mask)

    return mask


def mask_row_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every row y, the [left, right] extent of the drivable mask at
    that row. Rows with no evidence (e.g. near the vanishing point, or a
    stretch no vehicle ever crossed) get the nearest valid row's bounds
    carried forward/back, so every row has a usable, non-degenerate bound
    instead of silently falling back to the full frame width."""
    h, w = mask.shape
    left = np.full(h, -1.0)
    right = np.full(h, -1.0)
    for y in range(h):
        cols = np.flatnonzero(mask[y])
        if len(cols):
            left[y] = cols[0]
            right[y] = cols[-1]

    valid = left >= 0
    if not np.any(valid):
        # no evidence anywhere -- degrade gracefully to full frame rather
        # than crash; the caller's damped extrapolation is still in effect
        return np.zeros(h), np.full(h, w - 1.0)

    valid_idx = np.flatnonzero(valid)
    # forward-fill then back-fill missing rows from nearest valid row
    for arr in (left, right):
        last = valid_idx[0]
        for y in range(h):
            if valid[y]:
                last = y
            else:
                arr[y] = arr[last]
        last = valid_idx[-1]
        for y in range(h - 1, -1, -1):
            if valid[y]:
                last = y
            elif y < valid_idx[0]:
                arr[y] = arr[last]
    return left, right


def clip_x_to_mask(x: float, y: float, left_bounds: np.ndarray, right_bounds: np.ndarray, w: int) -> float:
    y_idx = int(np.clip(round(y), 0, len(left_bounds) - 1))
    lo, hi = left_bounds[y_idx], right_bounds[y_idx]
    if hi < lo:  # degenerate row, fall back to frame bounds
        lo, hi = 0, w - 1
    return float(np.clip(x, lo, hi))


# --------------------------------------------------------------------------
# Boundary + confidence
# --------------------------------------------------------------------------

def build_lane_model(
    curves: list[LaneCurve],
    frame_w: int,
    frame_h: int,
    roi_poly: Optional[np.ndarray],
    mask_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> dict:
    lanes = []
    y_lo = max(0, min(c.y_min for c in curves))
    y_hi = min(frame_h - 1, max(c.y_max for c in curves))
    sample_y = np.linspace(y_lo, y_hi, 40)

    centerlines = [c.eval(sample_y) for c in curves]
    spacings = [centerlines[i + 1] - centerlines[i] for i in range(len(centerlines) - 1)]
    left_bounds, right_bounds = mask_bounds if mask_bounds is not None else (None, None)

    for i, curve in enumerate(curves):
        center_x = centerlines[i]
        left_gap = spacings[i - 1] if i > 0 else (spacings[0] if spacings else np.full_like(center_x, 40))
        right_gap = spacings[i] if i < len(spacings) else (spacings[-1] if spacings else np.full_like(center_x, 40))
        left_x = center_x - left_gap / 2.0
        right_x = center_x + right_gap / 2.0
        left_x = np.clip(left_x, 0, frame_w - 1)
        right_x = np.clip(right_x, 0, frame_w - 1)

        # HARD backstop: clip every boundary point (and the centerline) to
        # the data-driven drivable-area mask for its row. This is what
        # stops the outer lanes (0 and k-1) from ballooning into the bus
        # shelter / sidewalk / fenced strip once the curve is extrapolated
        # past its real evidence -- the damped extrapolation controls the
        # *slope* out there, this controls the *absolute position*.
        if left_bounds is not None:
            for j, y in enumerate(sample_y):
                center_x[j] = clip_x_to_mask(center_x[j], y, left_bounds, right_bounds, frame_w)
                left_x[j] = clip_x_to_mask(left_x[j], y, left_bounds, right_bounds, frame_w)
                right_x[j] = clip_x_to_mask(right_x[j], y, left_bounds, right_bounds, frame_w)
            # re-enforce left <= center <= right after independent clipping
            left_x = np.minimum(left_x, center_x)
            right_x = np.maximum(right_x, center_x)

        centerline = [[float(x), float(y)] for x, y in zip(center_x, sample_y)]
        left_boundary = [[float(x), float(y)] for x, y in zip(left_x, sample_y)]
        right_boundary = [[float(x), float(y)] for x, y in zip(right_x, sample_y)]
        polygon = left_boundary + right_boundary[::-1]

        coverage = (curve.y_max - curve.y_min) / max(1.0, (y_hi - y_lo))
        fit_quality = 1.0 / (1.0 + curve.residual_px / 5.0)
        evidence_score = min(1.0, curve.n_points / 200.0)
        confidence = float(np.clip(0.4 * fit_quality + 0.35 * evidence_score + 0.25 * min(1.0, coverage), 0, 1))

        lanes.append({
            "lane_id": i,
            "direction": "unknown",
            "centerline": centerline,
            "left_boundary": left_boundary,
            "right_boundary": right_boundary,
            "polygon": polygon,
            "confidence": round(confidence, 3),
            "evidence": {
                "trajectories": round(evidence_score, 3),
                "geometry": round(fit_quality, 3),
                "coverage": round(coverage, 3),
            },
            "visible_fraction": round(coverage, 3),
            "support_y_range": [curve.y_min, curve.y_max],
        })
    return {"lanes": lanes, "sample_y_range": [float(y_lo), float(y_hi)]}


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

def draw_preview(frame: np.ndarray, lane_model: dict, out_path: str) -> None:
    overlay = frame.copy()
    colors = [(60, 180, 75), (0, 130, 200), (230, 25, 75), (245, 130, 48), (145, 30, 180), (70, 240, 240)]
    for lane in lane_model["lanes"]:
        poly = np.array(lane["polygon"], dtype=np.int32)
        color = colors[lane["lane_id"] % len(colors)]
        cv2.fillPoly(overlay, [poly], color)
    blended = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    for lane in lane_model["lanes"]:
        cl = np.array(lane["centerline"], dtype=np.int32)
        cv2.polylines(blended, [cl], False, (255, 255, 255), 2)
        mid = cl[len(cl) // 2]
        label = f'Lane {lane["lane_id"]}  conf {lane["confidence"]:.2f}'
        cv2.putText(blended, label, (int(mid[0]) - 40, int(mid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, blended)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--vehicle_classes", required=True,
                     help="comma-separated class ids to treat as vehicles, e.g. 1,3,4,5,6,13,14")
    ap.add_argument("--roi", default=None, help="optional road_roi.json from select_roi.py")
    ap.add_argument("--frame_index", type=int, default=500)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_lanes", type=int, default=6)
    ap.add_argument("--degree", type=int, default=1, choices=[1, 2])
    ap.add_argument("--no_drivable_mask", action="store_true",
                     help="disable the data-driven drivable-area mask and fall back to "
                          "damped extrapolation only (not recommended -- this is what "
                          "let Lane 0 balloon into the bus shelter)")
    ap.add_argument("--mask_close_px", type=int, default=41,
                     help="morphological close kernel size (bridges gaps between passing vehicles)")
    ap.add_argument("--mask_dilate_px", type=int, default=18,
                     help="morphological dilate kernel size (keeps mask generous, not lane-tight)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vehicle_classes = args.vehicle_classes.split(",")

    cfg = LaneFitConfig(vehicle_classes=vehicle_classes, max_lanes=args.max_lanes, degree=args.degree)

    print("[LOAD] reading trajectories...")
    tracks = load_tracks(args.csv, vehicle_classes)
    print(f"[LOAD] {len(tracks)} raw vehicle-only tracks")

    tracks = filter_tracks(tracks, cfg)
    print(f"[FILTER] {len(tracks)} tracks kept after length/span/straightness filtering")

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {args.frame_index} from {args.video}")
    h, w = frame.shape[:2]
    roi_poly = load_roi(args.roi, (h, w))

    mask_bounds = None
    if not args.no_drivable_mask:
        print("[MASK] building drivable-area mask from vehicle footprints...")
        drivable_mask = build_drivable_mask(
            args.csv, vehicle_classes, (h, w), roi_poly=roi_poly,
            close_px=args.mask_close_px, dilate_px=args.mask_dilate_px,
        )
        coverage_pct = 100.0 * drivable_mask.sum() / 255 / (h * w)
        print(f"[MASK] drivable area covers {coverage_pct:.1f}% of frame")
        mask_bounds = mask_row_bounds(drivable_mask)
        mask_debug_path = os.path.join(args.out_dir, "drivable_mask.jpg")
        mask_overlay = frame.copy()
        mask_overlay[drivable_mask > 0] = (
            0.5 * mask_overlay[drivable_mask > 0] + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)
        cv2.imwrite(mask_debug_path, mask_overlay)
        print(f"[SAVE] {mask_debug_path}  (inspect this if a lane still looks wrong)")

    print("[FIT] estimating lane count + fitting curves (EM, robust regression)...")
    curves, k = estimate_lane_count(tracks, cfg)
    if curves is None:
        print("LANE_MODEL_UNCERTAIN: no k in the configured range produced a "
              "non-crossing, sufficiently-supported fit. Lower --max_lanes, "
              "relax min_points_per_lane, or check vehicle_classes.")
        sys.exit(1)
    print(f"[FIT] settled on {len(curves)} lane(s)")
    for c in curves:
        print(f"  support y=[{c.y_min:.0f},{c.y_max:.0f}]  n={c.n_points}  "
              f"residual={c.residual_px:.2f}px")

    lane_model = build_lane_model(curves, w, h, roi_poly, mask_bounds=mask_bounds)
    lane_model["video"] = {"width": w, "height": h}
    lane_model["method"] = "automatic_hybrid_em_robust"
    lane_model["global_confidence"] = round(
        float(np.mean([l["confidence"] for l in lane_model["lanes"]])), 3
    )

    json_path = os.path.join(args.out_dir, "lane_model.json")
    with open(json_path, "w") as f:
        json.dump(lane_model, f, indent=2)
    print(f"[SAVE] {json_path}")

    preview_path = os.path.join(args.out_dir, "lane_preview_robust.jpg")
    draw_preview(frame, lane_model, preview_path)
    print(f"[SAVE] {preview_path}")


if __name__ == "__main__":
    main()
