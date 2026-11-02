"""
modules/lane_assigner.py

Vehicle-to-Lane Assignment, Debounce Hysteresis, and Traffic Violation Hooks.

Changes vs previous version:
- Scene-health (coverage) gate: WRONG_WAY / OUTSIDE / WEAVING only fire when the
  inferred lanes actually contain most active vehicles.
- Wrong-way ignores corridors whose direction is "mixed" (no clear consensus).
- OUTSIDE FOV guard uses perspective.road_roi_top_ratio from config.
- SPEEDING and signal rules (red light / stop line) are opt-in via config.
- Confirmed lane assignment now reverts to None when a vehicle leaves all lanes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from .temporal_lane_model import InferredLane
from .tracker_pipeline import TrackObservation, TrackSummary


@dataclass
class VehicleAssignment:
    track_id: int
    cls_name: str
    lane_id: Optional[int]
    lane_name: str
    lane_conf: float
    dist_from_center_px: float
    direction: str
    contact_point: Tuple[float, float]
    lane_position_status: str = "unknown"
    speed_kmh: float = 0.0
    is_violating: bool = False
    violation_types: List[str] = field(default_factory=list)


class LaneAssigner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        assign_cfg = config.get("lane_assignment", {})
        persp_cfg = config.get("perspective", {})

        self.lateral_tolerance = assign_cfg.get("lateral_tolerance_px", 25.0)
        self.wrong_way_angle = assign_cfg.get("wrong_way_angle_deg", 120.0)
        self.outside_min_lane_conf = assign_cfg.get("outside_min_lane_confidence", 0.85)
        self.outside_min_track_observations = assign_cfg.get("outside_min_track_observations", 15)
        self.outside_min_displacement_px = assign_cfg.get("outside_min_displacement_px", 40.0)
        self.outside_required_votes = assign_cfg.get("outside_required_votes", 8)
        self.outside_margin_multiplier = assign_cfg.get("outside_margin_multiplier", 3.0)
        self.min_violation_track_observations = assign_cfg.get("min_violation_track_observations", 20)
        self.min_violation_displacement_px = assign_cfg.get("min_violation_displacement_px", 60.0)

        # Scene-health gate
        self.min_coverage = float(assign_cfg.get("outside_min_coverage", 0.60))
        self.coverage_min_vehicles = int(assign_cfg.get("coverage_min_vehicles", 5))
        self.coverage_ema: Optional[float] = None  # None = not yet measured (untrusted)

        # ROI (shared with SceneAnalyzer / transformer)
        self.roi_top_ratio = float(persp_cfg.get("road_roi_top_ratio", 0.38))

        # Opt-in rule groups
        self.speed_enforcement_enabled = bool(assign_cfg.get("speed_enforcement_enabled", False))
        self.signal_rules_enabled = bool(assign_cfg.get("signal_rules_enabled", False))
        self.outside_road_enabled = bool(assign_cfg.get("outside_road_enabled", False))

        temp_cfg = config.get("temporal_model", {})
        self.debounce_len = temp_cfg.get("hysteresis_frames", 6)

        self.assignment_history: Dict[int, deque] = {}
        self.confirmed_assignments: Dict[int, Optional[int]] = {}
        self.lane_switch_counts: Dict[int, int] = {}
        self.outside_history: Dict[int, deque] = {}

        self.speed_limit_kmh = float(assign_cfg.get("speed_limit_kmh", 65.0))
        self.stop_line_y_ratio = float(assign_cfg.get("stop_line_y_ratio", 0.72))
        self.stop_line_tolerance_px = float(assign_cfg.get("stop_line_tolerance_px", 28.0))
        self.speed_history: Dict[int, deque] = {}

    # ------------------------------------------------------------------
    # Scene health
    # ------------------------------------------------------------------
    def _update_coverage(
        self,
        vehicles: List[TrackObservation],
        active_lanes: List[InferredLane],
    ) -> bool:
        """
        Updates the EMA of the fraction of vehicles whose contact point lies inside
        (or within tolerance of) an inferred lane. Returns True if lanes are trusted.
        """
        if not active_lanes:
            return False

        n = len(vehicles)
        if n >= self.coverage_min_vehicles:
            inside = 0
            for v in vehicles:
                pt = (float(v.bottom_center[0]), float(v.bottom_center[1]))
                for lane in active_lanes:
                    if cv2.pointPolygonTest(lane.polygon_img, pt, True) >= -self.lateral_tolerance:
                        inside += 1
                        break
            ratio = inside / n
            if self.coverage_ema is None:
                self.coverage_ema = ratio
            else:
                self.coverage_ema = 0.9 * self.coverage_ema + 0.1 * ratio

        return self.coverage_ema is not None and self.coverage_ema >= self.min_coverage

    # ------------------------------------------------------------------
    # Main assignment
    # ------------------------------------------------------------------
    def assign_vehicles(
        self,
        vehicles: List[TrackObservation],
        track_summaries: Dict[int, TrackSummary],
        active_lanes: List[InferredLane],
        corridors: Optional[List[Any]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        transformer: Optional[Any] = None,
        fps: float = 25.0,
        signal_phase: Optional[Any] = None,
        secondary_detections: Optional[List[Any]] = None,
    ) -> List[VehicleAssignment]:
        assignments: List[VehicleAssignment] = []

        if not active_lanes:
            for v in vehicles:
                assignments.append(
                    VehicleAssignment(
                        track_id=v.track_id,
                        cls_name=v.cls_name,
                        lane_id=None,
                        lane_name="No Active Lanes",
                        lane_conf=0.0,
                        dist_from_center_px=0.0,
                        direction="unknown",
                        contact_point=v.bottom_center,
                        lane_position_status="no_lane_model",
                    )
                )
            return assignments

        lanes_trusted = self._update_coverage(vehicles, active_lanes)

        for v in vehicles:
            tid = v.track_id
            contact_pt = (float(v.bottom_center[0]), float(v.bottom_center[1]))
            trk_sum = track_summaries.get(tid)
            trk_dir = trk_sum.direction if trk_sum else "inbound"

            # 1. Test against each lane polygon
            candidate_lane_id: Optional[int] = None
            all_dists: List[float] = []
            best_candidate_dist = -float("inf")
            min_dist_to_center = float("inf")

            for lane in active_lanes:
                dist = cv2.pointPolygonTest(lane.polygon_img, contact_pt, measureDist=True)
                all_dists.append(dist)
                if dist >= -self.lateral_tolerance and dist > best_candidate_dist:
                    best_candidate_dist = dist
                    candidate_lane_id = lane.lane_id

            best_dist = max(all_dists) if all_dists else -float("inf")

            # 2. Debounce hysteresis
            if tid not in self.assignment_history:
                self.assignment_history[tid] = deque(maxlen=self.debounce_len)
                self.confirmed_assignments[tid] = candidate_lane_id
                self.lane_switch_counts[tid] = 0
                self.outside_history[tid] = deque(maxlen=self.outside_required_votes + 2)

            self.assignment_history[tid].append(candidate_lane_id)
            votes = list(self.assignment_history[tid])
            current_confirmed = self.confirmed_assignments[tid]
            majority = max(2, len(votes) // 2 + 1)

            if candidate_lane_id is not None:
                if votes.count(candidate_lane_id) >= majority:
                    if current_confirmed is not None and current_confirmed != candidate_lane_id:
                        self.lane_switch_counts[tid] += 1
                    self.confirmed_assignments[tid] = candidate_lane_id
            else:
                # FIX: allow leaving all lanes (previously the old lane stuck forever)
                if votes.count(None) >= majority:
                    self.confirmed_assignments[tid] = None

            final_lane_id = self.confirmed_assignments[tid]

            # 3. Distance from centerline of assigned lane
            assigned_lane = None
            lane_name = "Outside Lane"
            lane_conf = 0.50
            lane_position_status = "outside_lane"
            if final_lane_id is not None:
                for lane in active_lanes:
                    if lane.lane_id == final_lane_id:
                        assigned_lane = lane
                        break
            if assigned_lane is not None:
                lane_name = assigned_lane.name
                lane_conf = assigned_lane.confidence
                lane_position_status = "in_lane"
                cl_pts = assigned_lane.centerline_img
                nearest_idx = int(np.argmin(np.abs(cl_pts[:, 1] - contact_pt[1])))
                min_dist_to_center = float(contact_pt[0] - cl_pts[nearest_idx][0])
            else:
                min_dist_to_center = 0.0

            # 4. Metric speed (only used for enforcement if enabled)
            speed_kmh = self._estimate_metric_speed(
                tid=tid, track_summary=trk_sum, transformer=transformer, fps=fps
            )

            # 5. Violations
            is_violating = False
            violations: List[str] = []
            enforcement_ready = self._is_track_enforcement_ready(trk_sum, trk_dir)

            # A. Wrong-way: needs trusted lanes AND a corridor with a clear direction
            if (
                lanes_trusted
                and enforcement_ready
                and assigned_lane is not None
                and trk_dir not in ("static", "unknown", "lateral")
            ):
                lane_dir = assigned_lane.direction
                if lane_dir == "inbound" and trk_dir == "outbound":
                    is_violating = True
                    violations.append("WRONG_WAY_DRIVING")
                elif lane_dir == "outbound" and trk_dir == "inbound":
                    is_violating = True
                    violations.append("WRONG_WAY_DRIVING")
                # lane_dir == "mixed" -> no wrong-way judgement

            # B. Speeding (opt-in, requires calibration)
            if (
                self.speed_enforcement_enabled
                and enforcement_ready
                and speed_kmh > self.speed_limit_kmh
                and trk_dir not in ("static", "unknown")
            ):
                is_violating = True
                violations.append("SPEEDING")

            # C. Weaving (depends on lane model quality)
            if lanes_trusted and enforcement_ready and self.lane_switch_counts.get(tid, 0) >= 3:
                is_violating = True
                violations.append("DANGEROUS_WEAVING")

            # D. Off-road / outside carriageway. Always reported as
            # lane_position_status for analytics; only ticketed when opted in
            # and the lane model is trusted.
            outside_candidate = self._is_reliable_outside_candidate(
                final_lane_id=final_lane_id,
                best_dist=best_dist,
                contact_pt=contact_pt,
                active_lanes=active_lanes,
                track_summary=trk_sum,
                track_direction=trk_dir,
                corridors=corridors,
                frame_shape=frame_shape,
            )
            self.outside_history.setdefault(tid, deque(maxlen=self.outside_required_votes + 2))
            self.outside_history[tid].append(bool(outside_candidate))
            outside_votes = sum(1 for vote in self.outside_history[tid] if vote)
            if outside_votes >= self.outside_required_votes:
                lane_position_status = "outside_drivable_corridor"
                if self.outside_road_enabled and lanes_trusted:
                    is_violating = True
                    violations.append("OUTSIDE_DRIVABLE_ROAD")

            # E. Signal breaches (opt-in, needs per-camera stop line)
            if self.signal_rules_enabled and signal_phase is not None and frame_shape is not None:
                fh, fw = frame_shape[:2]
                stop_line_y = fh * self.stop_line_y_ratio
                is_red = (
                    str(getattr(signal_phase, "phase_id", "")).startswith("red")
                    or not (getattr(signal_phase, "green_seconds", 0) > 0)
                    or "red" in str(getattr(signal_phase, "label", "")).lower()
                )
                if enforcement_ready and is_red and trk_dir == "inbound":
                    if contact_pt[1] > (stop_line_y + self.stop_line_tolerance_px):
                        if speed_kmh > 12.0 or (trk_sum and trk_sum.total_displacement > 35):
                            is_violating = True
                            violations.append("RED_LIGHT_VIOLATION")
                    elif abs(contact_pt[1] - stop_line_y) <= self.stop_line_tolerance_px:
                        if speed_kmh < 15.0:
                            is_violating = True
                            violations.append("STOP_LINE_ENCROACHMENT")

            # F. Emergency vehicle obstruction
            if secondary_detections:
                for sec in secondary_detections:
                    if getattr(sec, "detector", "") == "emergency_vehicle" or "ambulance" in str(
                        getattr(sec, "label", "")
                    ).lower():
                        amb_bbox = getattr(sec, "bbox", None)
                        if amb_bbox is not None:
                            amb_cy = (amb_bbox[1] + amb_bbox[3]) / 2.0
                            if (
                                contact_pt[1] > amb_cy
                                and abs(contact_pt[0] - (amb_bbox[0] + amb_bbox[2]) / 2.0) < 90
                            ):
                                if speed_kmh < 25.0 and v.cls_name not in ("bicycle", "person"):
                                    is_violating = True
                                    violations.append("EMERGENCY_OBSTRUCTION")

            # G. Helmet violation for two-wheelers
            if v.cls_name in ("motorcycle", "bicycle") and secondary_detections:
                for sec in secondary_detections:
                    if getattr(sec, "detector", "") == "helmet" and "no_helmet" in str(
                        getattr(sec, "label", "")
                    ).lower():
                        sec_box = getattr(sec, "bbox", None)
                        if sec_box and abs(sec_box[0] - v.bbox[0]) < 60:
                            is_violating = True
                            violations.append("NO_HELMET")

            assignments.append(
                VehicleAssignment(
                    track_id=tid,
                    cls_name=v.cls_name,
                    lane_id=final_lane_id,
                    lane_name=lane_name,
                    lane_conf=round(lane_conf, 2),
                    dist_from_center_px=round(min_dist_to_center, 1),
                    direction=trk_dir,
                    contact_point=contact_pt,
                    lane_position_status=lane_position_status,
                    speed_kmh=round(speed_kmh, 1),
                    is_violating=is_violating,
                    violation_types=violations,
                )
            )

        return assignments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_track_enforcement_ready(
        self,
        track_summary: Optional[TrackSummary],
        track_direction: str,
    ) -> bool:
        if track_summary is None:
            return False
        if len(track_summary.observations) < self.min_violation_track_observations:
            return False
        if track_summary.total_displacement < self.min_violation_displacement_px:
            return False
        if track_summary.is_parked or track_direction in ("static", "unknown"):
            return False
        return True

    def _is_reliable_outside_candidate(
        self,
        final_lane_id: Optional[int],
        best_dist: float,
        contact_pt: Tuple[float, float],
        active_lanes: List[InferredLane],
        track_summary: Optional[TrackSummary],
        track_direction: str,
        corridors: Optional[List[Any]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> bool:
        if final_lane_id is not None or not active_lanes:
            return False

        model_conf = float(np.mean([lane.confidence for lane in active_lanes]))
        if model_conf < self.outside_min_lane_conf:
            return False

        if track_summary is None:
            return False
        if len(track_summary.observations) < self.outside_min_track_observations:
            return False
        if track_summary.total_displacement < self.outside_min_displacement_px:
            return False
        if track_summary.is_parked or track_direction in ("static", "unknown"):
            return False

        # FOV guards: must match the lane ROI, otherwise the area above the ROI always looks "outside"
        if frame_shape is not None:
            fh, fw = frame_shape[:2]
            cx, cy = contact_pt
            if cy < fh * self.roi_top_ratio or cy > fh * 0.96:
                return False
            if cx < fw * 0.03 or cx > fw * 0.97:
                return False

        # Inside any detected corridor polygon => valid carriageway
        if corridors:
            for corridor in corridors:
                c_poly = getattr(corridor, "polygon", None)
                if c_poly is not None and len(c_poly) >= 3:
                    if cv2.pointPolygonTest(c_poly.astype(np.float32), contact_pt, False) >= 0:
                        return False

        outside_margin = self.lateral_tolerance * self.outside_margin_multiplier
        return best_dist < -outside_margin

    def _estimate_metric_speed(
        self,
        tid: int,
        track_summary: Optional[TrackSummary],
        transformer: Optional[Any],
        fps: float = 25.0,
    ) -> float:
        """
        Approximate speed (km/h). NOT calibrated: only used for enforcement when
        lane_assignment.speed_enforcement_enabled is true.
        """
        if track_summary is None or len(track_summary.observations) < 3:
            return 0.0
        if track_summary.is_parked or track_summary.direction in ("static", "unknown"):
            return 0.0

        window = min(len(track_summary.observations), 10)
        recent_obs = track_summary.observations[-window:]
        p_start = np.array(recent_obs[0].bottom_center)
        p_end = np.array(recent_obs[-1].bottom_center)
        frame_diff = max(1, recent_obs[-1].frame_idx - recent_obs[0].frame_idx)
        time_elapsed_sec = frame_diff / max(1.0, fps)

        if transformer is not None and getattr(transformer, "H", None) is not None:
            pts_img = np.array([p_start, p_end], dtype=np.float32)
            bev_pts = transformer.image_to_bev(pts_img)
            if len(bev_pts) == 2:
                meters_per_bev_px = float(
                    self.config.get("perspective", {}).get("meters_per_bev_pixel", 0.09)
                )
                bev_disp_px = float(np.linalg.norm(bev_pts[1] - bev_pts[0]))
                speed_kmh = (bev_disp_px * meters_per_bev_px / time_elapsed_sec) * 3.6
            else:
                speed_kmh = 0.0
        else:
            px_disp = float(np.linalg.norm(p_end - p_start))
            y_avg = (p_start[1] + p_end[1]) / 2.0
            scale_factor = 0.04 + (y_avg / 1080.0) * 0.06
            speed_kmh = (px_disp * scale_factor / time_elapsed_sec) * 3.6

        self.speed_history.setdefault(tid, deque(maxlen=8))
        self.speed_history[tid].append(speed_kmh)
        smooth_speed = float(np.median(self.speed_history[tid]))
        return max(0.0, min(180.0, smooth_speed))