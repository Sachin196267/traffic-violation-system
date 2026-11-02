"""
modules/temporal_lane_model.py

Persistent Lane Model with Exponential Moving Average (EMA) Temporal Smoothing.
Features:
- Maintains a persistent state of lane boundaries across video frames.
- Eliminates boundary fluttering and jitter via control-point EMA smoothing.
- Implements hysteresis on lane-count transitions (prevents abrupt 3->4->2 lane jumps).
- Manages occlusion grace periods so lane models don't vanish when road is occluded.
- Computes multi-source composite confidence scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .trajectory_clustering import InferredLane


@dataclass
class PersistentBoundary:
    boundary_id: int
    pts_img: np.ndarray      # (N, 2)
    pts_bev: np.ndarray      # (N, 2)
    alpha: float = 0.12

    def update(self, new_pts_img: np.ndarray, new_pts_bev: np.ndarray):
        if self.pts_img.shape == new_pts_img.shape:
            self.pts_img = (1.0 - self.alpha) * self.pts_img + self.alpha * new_pts_img
            self.pts_bev = (1.0 - self.alpha) * self.pts_bev + self.alpha * new_pts_bev
        else:
            self.pts_img = new_pts_img
            self.pts_bev = new_pts_bev


class TemporalLaneModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        temp_cfg = config.get("temporal_model", {})
        self.alpha = temp_cfg.get("ema_alpha", 0.12)
        self.hysteresis_frames = temp_cfg.get("hysteresis_frames", 6)
        self.min_conf = temp_cfg.get("min_confidence", 0.50)
        self.max_gap_frames = temp_cfg.get("max_gap_frames", 45)

        self.active_lanes: List[InferredLane] = []
        self.pending_lanes: Optional[List[InferredLane]] = None
        self.pending_count_sustained: int = 0
        self.last_update_frame: int = 0
        self.global_confidence: float = 0.50
        self.is_initialized: bool = False

    def update(self, new_lanes: List[InferredLane], frame_idx: int) -> List[InferredLane]:
        """
        Updates the persistent lane model with newly proposed lanes.
        Applies EMA smoothing if lane count is identical.
        Applies hysteresis if lane count proposes a change.
        """
        if not new_lanes:
            # Check occlusion grace period
            gap = frame_idx - self.last_update_frame
            if gap > self.max_gap_frames:
                self.global_confidence = max(0.20, self.global_confidence - 0.01)
            return self.active_lanes

        if not self.is_initialized:
            self.active_lanes = new_lanes
            self.is_initialized = True
            self.last_update_frame = frame_idx
            self.global_confidence = float(np.mean([l.confidence for l in new_lanes]))
            return self.active_lanes

        # Case 1: Same number of lanes -> Smooth control points
        if len(new_lanes) == len(self.active_lanes):
            self.pending_lanes = None
            self.pending_count_sustained = 0
            self.last_update_frame = frame_idx

            smoothed_lanes: List[InferredLane] = []
            for cur_lane, new_lane in zip(self.active_lanes, new_lanes):
                # Smooth boundaries
                sm_left_img = (1.0 - self.alpha) * cur_lane.left_boundary_img + self.alpha * new_lane.left_boundary_img
                sm_right_img = (1.0 - self.alpha) * cur_lane.right_boundary_img + self.alpha * new_lane.right_boundary_img
                sm_center_img = (1.0 - self.alpha) * cur_lane.centerline_img + self.alpha * new_lane.centerline_img

                sm_left_bev = (1.0 - self.alpha) * cur_lane.left_boundary_bev + self.alpha * new_lane.left_boundary_bev
                sm_right_bev = (1.0 - self.alpha) * cur_lane.right_boundary_bev + self.alpha * new_lane.right_boundary_bev
                sm_center_bev = (1.0 - self.alpha) * cur_lane.centerline_bev + self.alpha * new_lane.centerline_bev

                poly_img = np.vstack([sm_left_img, sm_right_img[::-1]]).astype(np.int32)
                sm_conf = (1.0 - self.alpha) * cur_lane.confidence + self.alpha * new_lane.confidence

                smoothed_lanes.append(
                    InferredLane(
                        lane_id=cur_lane.lane_id,
                        corridor_id=cur_lane.corridor_id,
                        name=cur_lane.name,
                        direction=cur_lane.direction,
                        left_boundary_bev=sm_left_bev,
                        right_boundary_bev=sm_right_bev,
                        centerline_bev=sm_center_bev,
                        left_boundary_img=sm_left_img,
                        right_boundary_img=sm_right_img,
                        centerline_img=sm_center_img,
                        polygon_img=poly_img,
                        avg_width_bev=float(cur_lane.avg_width_bev),
                        confidence=float(sm_conf),
                        supporting_tracks=new_lane.supporting_tracks,
                    )
                )
            self.active_lanes = smoothed_lanes
            self.global_confidence = float(np.mean([l.confidence for l in self.active_lanes]))
            return self.active_lanes

        # Case 2: Proposed lane count differs -> Require hysteresis
        if (
            self.pending_lanes is not None
            and len(self.pending_lanes) == len(new_lanes)
        ):
            self.pending_count_sustained += 1
            if self.pending_count_sustained >= self.hysteresis_frames:
                # Sustained long enough -> accept the transition
                self.active_lanes = new_lanes
                self.pending_lanes = None
                self.pending_count_sustained = 0
                self.last_update_frame = frame_idx
                self.global_confidence = float(np.mean([l.confidence for l in new_lanes]))
                return self.active_lanes
        else:
            self.pending_lanes = new_lanes
            self.pending_count_sustained = 1

        # Maintain prior stable model while waiting for hysteresis
        return self.active_lanes

    def export_boundaries_json(
        self, frame_w: int, frame_h: int, primary_only: bool = True
    ) -> Dict[str, Any]:
        """
        Exports boundaries in exact format compatible with lane_utils.py / verify_lanes.py:
        {"frame_width": W, "frame_height": H, "boundaries": [[[x, y], ...], ...]}
        """
        if not self.active_lanes:
            return {"frame_width": frame_w, "frame_height": frame_h, "boundaries": []}

        lanes_to_export = [
            l for l in self.active_lanes if l.corridor_id == 0
        ] if primary_only else self.active_lanes
        if not lanes_to_export:
            lanes_to_export = self.active_lanes

        boundaries_list = []
        # First left boundary
        b0 = [[float(pt[0]), float(pt[1])] for pt in lanes_to_export[0].left_boundary_img]
        boundaries_list.append(b0)

        # Subsequent right boundaries
        for lane in lanes_to_export:
            b = [[float(pt[0]), float(pt[1])] for pt in lane.right_boundary_img]
            boundaries_list.append(b)

        return {
            "frame_width": frame_w,
            "frame_height": frame_h,
            "num_lanes": len(lanes_to_export),
            "global_confidence": round(self.global_confidence, 3),
            "boundaries": boundaries_list,
        }
