"""
modules/visualizer.py

Multi-Layer Visualization Engine for Automatic Lane Detection & Vehicle Assignment.
Features:
- Semi-transparent shaded lane corridors with distinct palette colors.
- High-contrast boundary lines and centerlines.
- On-road perspective lane labels ("LANE 1", "LANE 2", etc.).
- Vehicle bounding boxes with badge cards: Track ID, Class, Assigned Lane, Confidence.
- Vehicle tire-road contact point markers.
- Picture-in-picture (PiP) Bird's-Eye Debug View showing BEV trajectories, boundaries, and divider.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from .lane_assigner import VehicleAssignment
from .perspective_transformer import PerspectiveTransformer
from .scene_analyzer import RoadCorridor, DividerBoundary
from .temporal_lane_model import InferredLane
from .tracker_pipeline import TrackObservation, TrackSummary


LANE_PALETTE = [
    (240, 120, 20),   # Lane 1: Blue-ish / Teal
    (40, 190, 80),    # Lane 2: Emerald Green
    (20, 160, 240),   # Lane 3: Amber / Orange
    (180, 50, 220),   # Lane 4: Purple
    (220, 220, 40),   # Lane 5: Cyan
]


class LaneVisualizer:
    def __init__(self, config: Dict[str, Any], frame_w: int, frame_h: int):
        self.config = config
        self.frame_w = frame_w
        self.frame_h = frame_h

        vis_cfg = config.get("visualization", {})
        self.fill_alpha = vis_cfg.get("fill_alpha", 0.25)
        self.line_thick = vis_cfg.get("line_thickness", 3)
        self.show_contact = vis_cfg.get("show_contact_points", True)
        self.show_trails = vis_cfg.get("show_trajectories", True)
        self.show_birdseye_hud = vis_cfg.get("show_birdseye_hud", True)
        self.hud_w = vis_cfg.get("birdseye_hud_w", 320)
        self.hud_h = vis_cfg.get("birdseye_hud_h", 360)

    def render_frame(
        self,
        frame: np.ndarray,
        active_lanes: List[InferredLane],
        dividers: List[DividerBoundary],
        assignments: List[VehicleAssignment],
        current_vehicles: List[TrackObservation],
        track_summaries: Dict[int, TrackSummary],
        transformer: PerspectiveTransformer,
        frame_idx: int,
        fps: float = 25.0,
    ) -> np.ndarray:
        """
        Renders complete multi-layer visualization onto the video frame.
        """
        out = frame.copy()
        overlay = frame.copy()

        # 1. Draw Physical Dividers / Medians (Yellow / Black hatched pattern or amber shading)
        for div in dividers:
            if div.polygon is not None:
                cv2.fillPoly(overlay, [div.polygon], (40, 40, 40))
                cv2.polylines(out, [div.polygon], True, (0, 200, 255), 2, cv2.LINE_AA)

        y_top_thresh = int(self.frame_h * self.config.get("perspective", {}).get("road_roi_top_ratio", 0.41))

        # 2. Draw Shaded Lane Polygons & Inferred Boundaries
        for i, lane in enumerate(active_lanes):
            color = LANE_PALETTE[i % len(LANE_PALETTE)]
            if lane.polygon_img is not None and len(lane.polygon_img) > 2:
                clipped_poly = lane.polygon_img[lane.polygon_img[:, 1] >= y_top_thresh]
                if len(clipped_poly) > 2:
                    cv2.fillPoly(overlay, [clipped_poly], color)

        # Blend shaded polygons
        if self.fill_alpha > 0:
            cv2.addWeighted(overlay, self.fill_alpha, out, 1.0 - self.fill_alpha, 0, out)

        # 3. Draw Lane Boundaries & Centerlines
        for i, lane in enumerate(active_lanes):
            # Left boundary (clipped to drivable road region)
            pts_l = lane.left_boundary_img.astype(np.int32)
            pts_l = pts_l[(pts_l[:, 1] >= y_top_thresh) & (pts_l[:, 1] <= self.frame_h)]
            if len(pts_l) >= 2:
                cv2.polylines(out, [pts_l], False, (255, 255, 255), self.line_thick, cv2.LINE_AA)

            # Right boundary (clipped to drivable road region)
            pts_r = lane.right_boundary_img.astype(np.int32)
            pts_r = pts_r[(pts_r[:, 1] >= y_top_thresh) & (pts_r[:, 1] <= self.frame_h)]
            if len(pts_r) >= 2:
                cv2.polylines(out, [pts_r], False, (255, 255, 255), self.line_thick, cv2.LINE_AA)

            # Dashed Centerline
            pts_c = lane.centerline_img.astype(np.int32)
            pts_c = pts_c[(pts_c[:, 1] >= y_top_thresh) & (pts_c[:, 1] <= self.frame_h)]
            for j in range(0, len(pts_c) - 1, 4):
                p1 = tuple(pts_c[j])
                p2 = tuple(pts_c[min(j + 2, len(pts_c) - 1)])
                cv2.line(out, p1, p2, (200, 240, 255), 1, cv2.LINE_AA)

            # Draw "LANE X" label on road near bottom
            if len(pts_c) > 10:
                label_pt = tuple(pts_c[int(len(pts_c) * 0.75)])
                label_text = f"LANE {lane.lane_id + 1}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
                lx, ly = label_pt[0] - tw // 2, label_pt[1]
                # Label backdrop badge
                cv2.rectangle(out, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 6), (20, 20, 20), -1)
                cv2.rectangle(out, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 6), LANE_PALETTE[i % len(LANE_PALETTE)], 2)
                cv2.putText(out, label_text, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # 4. Draw Vehicle Bounding Boxes and Assignment Cards
        assign_map = {a.track_id: a for a in assignments}

        for v in current_vehicles:
            tid = v.track_id
            assign = assign_map.get(tid)
            x1, y1, x2, y2 = [int(c) for c in v.bbox]

            # Box color: Red if violation, else by assigned lane
            if assign and assign.is_violating:
                box_color = (0, 0, 255)
            elif assign and assign.lane_id is not None:
                box_color = LANE_PALETTE[assign.lane_id % len(LANE_PALETTE)]
            else:
                box_color = (180, 180, 180)

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

            # Contact point marker
            if self.show_contact:
                cx, cy = int(v.bottom_center[0]), int(v.bottom_center[1])
                cv2.circle(out, (cx, cy), 4, (0, 255, 255), -1)
                cv2.circle(out, (cx, cy), 6, (0, 0, 0), 1)

            # Recent trajectory trail (filter out tracker ID jumps)
            if self.show_trails and tid in track_summaries:
                pts = track_summaries[tid].history_bottom_centers[-15:]
                if len(pts) >= 2:
                    clean_pts = [pts[0]]
                    for k in range(1, len(pts)):
                        dist = np.hypot(pts[k][0] - pts[k-1][0], pts[k][1] - pts[k-1][1])
                        if dist < 65.0:
                            clean_pts.append(pts[k])
                        else:
                            clean_pts = [pts[k]]
                    if len(clean_pts) >= 2:
                        trail_pts = np.array(clean_pts, dtype=np.int32)
                        cv2.polylines(out, [trail_pts], False, box_color, 2, cv2.LINE_AA)

            # Info Badge Card
            lane_str = f"L{assign.lane_id + 1}" if (assign and assign.lane_id is not None) else "OUT"
            conf_str = f"{assign.lane_conf:.2f}" if assign else f"{v.confidence:.2f}"
            badge_text = f"#{tid} {v.cls_name[:4].upper()} [{lane_str}|{conf_str}]"
            if assign and assign.is_violating:
                badge_text += f" !{assign.violation_types[0][:7]}!"

            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            by = max(y1, th + 8)
            cv2.rectangle(out, (x1, by - th - 5), (x1 + tw + 4, by + 3), (25, 25, 25), -1)
            cv2.rectangle(out, (x1, by - th - 5), (x1 + tw + 4, by + 3), box_color, 1)
            cv2.putText(out, badge_text, (x1 + 2, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Top HUD Summary Overlay
        self._render_hud_banner(out, active_lanes, assignments, frame_idx, fps)

        # 6. Picture-in-Picture Bird's-Eye Debug View
        if self.show_birdseye_hud:
            bev_canvas = self.render_birdseye_canvas(
                active_lanes, dividers, track_summaries, transformer
            )
            # Resize BEV canvas to HUD dimensions
            hud_bev = cv2.resize(bev_canvas, (self.hud_w, self.hud_h))
            # Border around PiP
            cv2.rectangle(hud_bev, (0, 0), (self.hud_w - 1, self.hud_h - 1), (0, 200, 255), 2)
            cv2.putText(
                hud_bev, "BIRD'S-EYE VIEW (IPM)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
            )

            # Place in upper right corner with small margin
            px = self.frame_w - self.hud_w - 20
            py = 65
            out[py:py + self.hud_h, px:px + self.hud_w] = hud_bev

        return out

    def _render_hud_banner(
        self,
        frame: np.ndarray,
        active_lanes: List[InferredLane],
        assignments: List[VehicleAssignment],
        frame_idx: int,
        fps: float,
    ):
        """Draws clean analytics banner at the top of the video."""
        banner_h = 55
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_w, banner_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, banner_h), (self.frame_w, banner_h), (50, 60, 70), 2)

        # Title & System Status
        title = "ROBUST AUTOMATIC LANE SYSTEM (INDIAN TRAFFIC)"
        cv2.putText(frame, title, (20, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        n_lanes = len(active_lanes)
        conf_avg = np.mean([l.confidence for l in active_lanes]) if active_lanes else 0.0
        n_tracked = len(assignments)
        n_viol = sum(1 for a in assignments if a.is_violating)

        metrics = (
            f"Frame: {frame_idx:04d} | Inferred Lanes: {n_lanes} | "
            f"Model Conf: {conf_avg:.2f} | Active Vehicles: {n_tracked} | "
            f"Violations: {n_viol}"
        )
        cv2.putText(frame, metrics, (20, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)

    def render_birdseye_canvas(
        self,
        active_lanes: List[InferredLane],
        dividers: List[DividerBoundary],
        track_summaries: Dict[int, TrackSummary],
        transformer: PerspectiveTransformer,
    ) -> np.ndarray:
        """
        Creates a dedicated top-down bird's-eye debug view.
        """
        canvas = np.zeros((transformer.ipm_h, transformer.ipm_w, 3), dtype=np.uint8)
        canvas[:] = (30, 32, 35)  # Dark asphalt background

        # Draw lateral grid lines (every 100 px)
        for y in range(0, transformer.ipm_h, 100):
            cv2.line(canvas, (0, y), (transformer.ipm_w, y), (45, 48, 52), 1)

        # 1. Shaded Lanes in BEV
        overlay = canvas.copy()
        for i, lane in enumerate(active_lanes):
            color = LANE_PALETTE[i % len(LANE_PALETTE)]
            pts_l = lane.left_boundary_bev
            pts_r = lane.right_boundary_bev
            poly = np.vstack([pts_l, pts_r[::-1]]).astype(np.int32)
            cv2.fillPoly(overlay, [poly], color)

        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)

        # 2. Draw Boundaries & Centerlines
        for i, lane in enumerate(active_lanes):
            pts_l = lane.left_boundary_bev.astype(np.int32)
            pts_r = lane.right_boundary_bev.astype(np.int32)
            pts_c = lane.centerline_bev.astype(np.int32)

            cv2.polylines(canvas, [pts_l], False, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.polylines(canvas, [pts_r], False, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.polylines(canvas, [pts_c], False, (200, 240, 255), 1, cv2.LINE_AA)

            cx = int(lane.centerline_bev[len(lane.centerline_bev) // 2, 0])
            cy = int(transformer.ipm_h * 0.85)
            cv2.putText(
                canvas, f"L{i+1}", (cx - 12, cy),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )

        # 3. Draw Vehicle Contact Trails in BEV
        for tid, track in track_summaries.items():
            pts_img = np.array(track.history_bottom_centers[-30:])
            if len(pts_img) < 2:
                continue
            pts_bev = transformer.image_to_bev(pts_img).astype(np.int32)
            # Clip to canvas
            valid = (
                (pts_bev[:, 0] >= 0) & (pts_bev[:, 0] < transformer.ipm_w) &
                (pts_bev[:, 1] >= 0) & (pts_bev[:, 1] < transformer.ipm_h)
            )
            if np.sum(valid) >= 2:
                color = (0, 255, 255) if not track.is_parked else (100, 100, 100)
                cv2.polylines(canvas, [pts_bev[valid]], False, color, 2, cv2.LINE_AA)
                latest_pt = tuple(pts_bev[valid][-1])
                cv2.circle(canvas, latest_pt, 4, (0, 0, 255), -1)

        return canvas
