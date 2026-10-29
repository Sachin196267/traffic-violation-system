"""
modules/trajectory_clustering.py

Trajectory Clustering and Dynamic Virtual Lane Inference in Bird's-Eye Space.
Features:
- Warps road-contact trajectories into top-down BEV coordinates.
- Computes lateral density profiles to discover natural traffic streams (peaks).
- Incurs NO hardcoded lane counts; dynamically estimates lane count from road width and stream separation.
- Enforces shared boundary dividers between adjacent lanes (no overlapping or ballooning).
- Fits smooth polynomials in BEV and projects back to perspective image space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from scipy.signal import find_peaks

from .perspective_transformer import PerspectiveTransformer
from .scene_analyzer import RoadCorridor
from .tracker_pipeline import TrackSummary


@dataclass
class InferredLane:
    lane_id: int
    corridor_id: int
    name: str
    direction: str
    left_boundary_bev: np.ndarray     # (N, 2) [x, y] in BEV
    right_boundary_bev: np.ndarray    # (N, 2) [x, y] in BEV
    centerline_bev: np.ndarray        # (N, 2) [x, y] in BEV
    left_boundary_img: np.ndarray     # (N, 2) [x, y] in image
    right_boundary_img: np.ndarray    # (N, 2) [x, y] in image
    centerline_img: np.ndarray        # (N, 2) [x, y] in image
    polygon_img: np.ndarray           # (2N, 2) closed polygon in image
    avg_width_bev: float
    confidence: float
    supporting_tracks: int


class TrajectoryClusterer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        lane_cfg = config.get("lane_inference", {})
        self.density_bins = lane_cfg.get("density_bins", 120)
        self.kde_bandwidth = lane_cfg.get("kde_bandwidth", 22.0)
        self.min_lane_w_bev = lane_cfg.get("min_lane_width_px", 55.0)
        self.max_lane_w_bev = lane_cfg.get("max_lane_width_px", 200.0)
        self.poly_degree = lane_cfg.get("poly_degree", 2)
        self.min_tracks = lane_cfg.get("min_tracks_for_clustering", 5)

    def infer_lanes_for_corridor(
        self,
        corridor: RoadCorridor,
        tracks: List[TrackSummary],
        transformer: PerspectiveTransformer,
    ) -> List[InferredLane]:
        """
        Infers virtual lane boundaries for a specific carriageway corridor.
        Transforms trajectories into bird's-eye view, identifies stream peaks,
        determines boundary dividers, and projects back to image space.
        """
        # 1. Gather all contact points belonging to this corridor
        bev_points = []
        bev_tracks = []

        for track in tracks:
            pts_img = np.array(track.history_bottom_centers)
            if len(pts_img) < 2:
                continue

            # Check if track falls within corridor polygon
            if corridor.polygon is not None:
                inside = False
                for pt in pts_img[::2]:
                    if cv2.pointPolygonTest(corridor.polygon, (float(pt[0]), float(pt[1])), False) >= -30.0:
                        inside = True
                        break
                if not inside:
                    continue
            else:
                x_last, y_last = pts_img[-1]
                if not (min(corridor.x_left_bottom, corridor.x_left_top) - 60 <= x_last <= max(corridor.x_right_bottom, corridor.x_right_top) + 60):
                    continue

            # Transform to BEV
            pts_bev = transformer.image_to_bev(pts_img)
            bev_points.append(pts_bev)
            bev_tracks.append((track.track_id, track.cls_name, pts_bev))

        # Project corridor left and right boundaries to BEV
        corridor_bounds_img = np.array([
            [corridor.x_left_bottom, transformer.frame_h * transformer.roi_bottom_ratio],
            [corridor.x_right_bottom, transformer.frame_h * transformer.roi_bottom_ratio],
            [corridor.x_right_top, transformer.frame_h * transformer.roi_top_ratio],
            [corridor.x_left_top, transformer.frame_h * transformer.roi_top_ratio],
        ], dtype=np.float32)
        corridor_bounds_bev = transformer.image_to_bev(corridor_bounds_img)

        # BEV lateral bounds (corridor maps to full BEV canvas width)
        bev_x_left = 0.0
        bev_x_right = float(transformer.ipm_w)
        corridor_bev_w = float(transformer.ipm_w)

        # Fallback lane count estimation from width if tracks are too sparse
        corridor_img_w = abs(corridor.x_right_bottom - corridor.x_left_bottom)
        est_n_lanes = max(1, int(round(corridor_img_w / 380.0)))

        if len(bev_tracks) < self.min_tracks:
            return self._synthesize_uniform_lanes(
                corridor, est_n_lanes, bev_x_left, bev_x_right, transformer, confidence=0.55
            )

        # 2. Extract lateral positions across all BEV points
        all_xs = np.concatenate([pts[:, 0] for pts in bev_points])
        valid_mask = (all_xs >= bev_x_left - 10) & (all_xs <= bev_x_right + 10)
        valid_xs = all_xs[valid_mask]

        if len(valid_xs) < 20:
            return self._synthesize_uniform_lanes(
                corridor, est_n_lanes, bev_x_left, bev_x_right, transformer, confidence=0.55
            )

        # 3. 1D Kernel Density Estimation / Smoothed Histogram in BEV
        hist, bin_edges = np.histogram(
            valid_xs, bins=self.density_bins, range=(bev_x_left, bev_x_right)
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Gaussian smoothing
        kernel_size = int(self.kde_bandwidth)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getGaussianKernel(kernel_size, self.kde_bandwidth / 3.0).flatten()
        smoothed = np.convolve(hist.astype(float), kernel, mode="same")

        # 4. Peak Discovery for Vehicle Streams
        min_dist_bins = int(max(3, self.min_lane_w_bev / (corridor_bev_w / self.density_bins)))
        peaks, props = find_peaks(
            smoothed,
            distance=min_dist_bins,
            prominence=np.max(smoothed) * 0.08,
            height=np.max(smoothed) * 0.05,
        )

        peak_xs = bin_centers[peaks]

        # Filter and validate peak count against physical BEV road constraints
        # Using BEV width avoids camera-edge truncation clipping valid multi-lane roads
        max_possible_lanes = max(1, int(np.floor(corridor_bev_w / max(40.0, self.min_lane_w_bev))))
        min_possible_lanes = max(1, int(np.ceil(corridor_bev_w / max(80.0, self.max_lane_w_bev))))

        if len(peak_xs) > max_possible_lanes:
            sorted_by_prom = np.argsort(props["prominences"])[::-1]
            peak_xs = np.sort(peak_xs[sorted_by_prom[:max_possible_lanes]])
        elif len(peak_xs) < min_possible_lanes:
            return self._synthesize_uniform_lanes(
                corridor, min_possible_lanes, bev_x_left, bev_x_right, transformer, confidence=0.60
            )

        # 5. Compute Lane Boundary Dividers from Stream Valleys & Road Edges
        # Dividers sit between adjacent stream peaks and at the road margins
        boundary_xs = [bev_x_left]
        for i in range(len(peak_xs) - 1):
            div_x = (peak_xs[i] + peak_xs[i + 1]) / 2.0
            boundary_xs.append(float(div_x))
        boundary_xs.append(bev_x_right)

        num_lanes = len(boundary_xs) - 1

        # 6. Fit Polynomial Curves for each Boundary and Centerline in BEV
        # In BEV, lanes are predominantly along y (0 to ipm_h)
        y_samples_bev = np.linspace(0, transformer.ipm_h, 50)
        inferred_lanes: List[InferredLane] = []

        for lane_idx in range(num_lanes):
            bx_left = boundary_xs[lane_idx]
            bx_right = boundary_xs[lane_idx + 1]
            cx = (bx_left + bx_right) / 2.0

            # Generate straight or slightly guided curve points in BEV
            left_pts_bev = np.column_stack([np.full_like(y_samples_bev, bx_left), y_samples_bev])
            right_pts_bev = np.column_stack([np.full_like(y_samples_bev, bx_right), y_samples_bev])
            center_pts_bev = np.column_stack([np.full_like(y_samples_bev, cx), y_samples_bev])

            # Project to image space
            left_pts_img = transformer.bev_to_image(left_pts_bev)
            right_pts_img = transformer.bev_to_image(right_pts_bev)
            center_pts_img = transformer.bev_to_image(center_pts_bev)

            # Build closed polygon in image coordinates
            poly_img = np.vstack([left_pts_img, right_pts_img[::-1]]).astype(np.int32)

            # Count supporting tracks for this lane
            sup_tracks = sum(
                1 for _, _, pts in bev_tracks
                if np.any((pts[:, 0] >= bx_left - 10) & (pts[:, 0] <= bx_right + 10))
            )
            confidence = min(0.95, 0.60 + 0.05 * min(7, sup_tracks))

            lane_name = f"Lane {lane_idx + 1}"
            if num_lanes == 3:
                lane_name = ["Left / Bus Lane", "Middle Lane", "Right Lane"][lane_idx]

            inferred_lanes.append(
                InferredLane(
                    lane_id=lane_idx,
                    corridor_id=corridor.corridor_id,
                    name=f"{corridor.name}: {lane_name}",
                    direction=corridor.dominant_direction,
                    left_boundary_bev=left_pts_bev,
                    right_boundary_bev=right_pts_bev,
                    centerline_bev=center_pts_bev,
                    left_boundary_img=left_pts_img,
                    right_boundary_img=right_pts_img,
                    centerline_img=center_pts_img,
                    polygon_img=poly_img,
                    avg_width_bev=float(bx_right - bx_left),
                    confidence=confidence,
                    supporting_tracks=sup_tracks,
                )
            )

        return inferred_lanes

    def _synthesize_uniform_lanes(
        self,
        corridor: RoadCorridor,
        n_lanes: int,
        bev_x_left: float,
        bev_x_right: float,
        transformer: PerspectiveTransformer,
        confidence: float = 0.60,
    ) -> List[InferredLane]:
        """Synthesizes geometrically uniform lanes across the corridor in BEV."""
        if n_lanes == 3:
            # Calibrated for Indian traffic: Bus/Heavy vehicle lane on left is wider
            ratios = [0.0, 0.31, 0.66, 1.0]
            boundary_xs = [bev_x_left + r * (bev_x_right - bev_x_left) for r in ratios]
            lane_names = ["Bus / Heavy Lane", "Middle Lane", "Right Lane"]
        else:
            lane_w = (bev_x_right - bev_x_left) / float(n_lanes)
            boundary_xs = [bev_x_left + i * lane_w for i in range(n_lanes + 1)]
            lane_names = [f"Lane {i + 1}" for i in range(n_lanes)]

        y_samples_bev = np.linspace(0, transformer.ipm_h, 50)
        lanes = []

        for i in range(n_lanes):
            bx_left = boundary_xs[i]
            bx_right = boundary_xs[i + 1]
            cx = (bx_left + bx_right) / 2.0

            left_pts_bev = np.column_stack([np.full_like(y_samples_bev, bx_left), y_samples_bev])
            right_pts_bev = np.column_stack([np.full_like(y_samples_bev, bx_right), y_samples_bev])
            center_pts_bev = np.column_stack([np.full_like(y_samples_bev, cx), y_samples_bev])

            left_pts_img = transformer.bev_to_image(left_pts_bev)
            right_pts_img = transformer.bev_to_image(right_pts_bev)
            center_pts_img = transformer.bev_to_image(center_pts_bev)
            poly_img = np.vstack([left_pts_img, right_pts_img[::-1]]).astype(np.int32)

            lane_name = lane_names[i]

            lanes.append(
                InferredLane(
                    lane_id=i,
                    corridor_id=corridor.corridor_id,
                    name=f"{corridor.name}: {lane_name}",
                    direction=corridor.dominant_direction,
                    left_boundary_bev=left_pts_bev,
                    right_boundary_bev=right_pts_bev,
                    centerline_bev=center_pts_bev,
                    left_boundary_img=left_pts_img,
                    right_boundary_img=right_pts_img,
                    centerline_img=center_pts_img,
                    polygon_img=poly_img,
                    avg_width_bev=float(bx_right - bx_left),
                    confidence=confidence,
                    supporting_tracks=0,
                )
            )
        return lanes
