#!/usr/bin/env python3
"""
Digital Twin Rendering Engine.
Computes planar homography perspective mapping from camera video to top-down/2.5D
synthetic digital twin environment with vehicle avatars, trajectory trails, and signal HUD.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class DigitalTwinEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        dt_cfg = config.get("digital_twin", {})
        self.canvas_w = dt_cfg.get("canvas_width", 600)
        self.canvas_h = dt_cfg.get("canvas_height", 720)
        self.road_color = tuple(dt_cfg.get("road_color", [35, 38, 46]))
        self.lane_color = tuple(dt_cfg.get("lane_marking_color", [255, 215, 0]))
        self.curb_color = tuple(dt_cfg.get("curb_color", [75, 85, 99]))

        # Homography calibration
        homo_cfg = config.get("spatial", {}).get("homography", {})
        src = np.array(homo_cfg.get("src_points", [
            [380, 320], [720, 320], [980, 620], [160, 620]
        ]), dtype=np.float32)
        dst = np.array(homo_cfg.get("dst_points", [
            [150, 80], [450, 80], [450, 580], [150, 580]
        ]), dtype=np.float32)

        self.H = cv2.getPerspectiveTransform(src, dst)
        self.H_inv = cv2.getPerspectiveTransform(dst, src)

        # Track trajectory history: {track_id: [(x_twin, y_twin, timestamp_frame)]}
        self.trajectories: Dict[int, List[Tuple[float, float, int]]] = {}
        self.max_trail_len = 35

    def project_point(self, pt: Tuple[float, float]) -> Tuple[int, int]:
        """Project a single camera point (x, y) to digital twin coordinates."""
        p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(p, self.H)
        tx, ty = transformed[0][0]
        return int(np.clip(tx, 0, self.canvas_w - 1)), int(np.clip(ty, 0, self.canvas_h - 1))

    def project_polygon(self, pts: List[List[int]]) -> np.ndarray:
        """Project camera polygon points to twin canvas."""
        p_arr = np.array([[pt for pt in pts]], dtype=np.float32)
        trans = cv2.perspectiveTransform(p_arr, self.H)
        return np.int32(trans[0])

    def render_base_canvas(self, signal_state: str, signal_countdown: float) -> np.ndarray:
        """Render static synthetic roadway, lane corridor, curbs, and traffic light HUD."""
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        canvas[:] = (20, 24, 33)  # Dark tech background

        # Draw main road surface
        cv2.rectangle(canvas, (100, 40), (self.canvas_w - 100, self.canvas_h - 40), self.road_color, -1)

        # Road boundaries / curbs
        cv2.line(canvas, (100, 40), (100, self.canvas_h - 40), self.curb_color, 4)
        cv2.line(canvas, (self.canvas_w - 100, 40), (self.canvas_w - 100, self.canvas_h - 40), self.curb_color, 4)

        # Center dashed dividers
        mid_x = self.canvas_w // 2
        dash_len = 25
        gap = 18
        for y in range(50, self.canvas_h - 50, dash_len + gap):
            cv2.line(canvas, (mid_x, y), (mid_x, min(y + dash_len, self.canvas_h - 50)), (180, 180, 180), 2)

        # Lane 1 & 2 dividing dashes
        for x_offset in [mid_x - 75, mid_x + 75]:
            for y in range(50, self.canvas_h - 50, dash_len + gap):
                cv2.line(canvas, (x_offset, y), (x_offset, min(y + dash_len, self.canvas_h - 50)), (100, 110, 125), 1)

        # Virtual Lane Corridor Highlight
        poly_cam = self.config.get("spatial", {}).get("virtual_lane_polygon", [])
        if poly_cam:
            poly_twin = self.project_polygon(poly_cam)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [poly_twin], (255, 210, 0))  # Semi-transparent cyan
            canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
            cv2.polylines(canvas, [poly_twin], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # Virtual Stop Line
        stop_line_cfg = self.config.get("spatial", {}).get("stop_line", {})
        p1 = tuple(stop_line_cfg.get("p1", [250, 490]))
        p2 = tuple(stop_line_cfg.get("p2", [900, 490]))
        p1_twin = self.project_point(p1)
        p2_twin = self.project_point(p2)
        stop_line_color = (0, 0, 255) if signal_state == "RED" else (255, 255, 255)
        cv2.line(canvas, (100, p1_twin[1]), (self.canvas_w - 100, p2_twin[1]), stop_line_color, 4, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "STOP LINE", (110, p1_twin[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, stop_line_color, 1, cv2.LINE_AA)

        # Header Title
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, 40), (14, 18, 25), -1)
        cv2.putText(canvas, "3D DIGITAL TWIN & REAL-TIME REPLICA", (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 240, 255), 2, cv2.LINE_AA)

        # Traffic Signal Light Head (Top right of canvas)
        self._render_signal_head(canvas, signal_state, signal_countdown, self.canvas_w - 90, 8)

        return canvas

    def _render_signal_head(self, canvas: np.ndarray, state: str, countdown: float, x: int, y: int):
        """Render a 3-aspect traffic light fixture with countdown timer."""
        # Signal Housing
        cv2.rectangle(canvas, (x, y), (x + 80, y + 25), (25, 30, 40), -1)
        cv2.rectangle(canvas, (x, y), (x + 80, y + 25), (70, 80, 95), 1)

        # 3 Circles: Red, Yellow, Green
        # Red lamp
        r_col = (0, 0, 255) if state == "RED" else (20, 20, 70)
        cv2.circle(canvas, (x + 14, y + 13), 8, r_col, -1)
        # Yellow lamp
        y_col = (0, 215, 255) if state == "YELLOW" else (20, 60, 70)
        cv2.circle(canvas, (x + 35, y + 13), 8, y_col, -1)
        # Green lamp
        g_col = (0, 255, 0) if state == "GREEN" else (20, 70, 20)
        cv2.circle(canvas, (x + 56, y + 13), 8, g_col, -1)

        # Glowing ring on active lamp
        if state == "RED":
            cv2.circle(canvas, (x + 14, y + 13), 10, (0, 0, 255), 1, cv2.LINE_AA)
        elif state == "YELLOW":
            cv2.circle(canvas, (x + 35, y + 13), 10, (0, 215, 255), 1, cv2.LINE_AA)
        elif state == "GREEN":
            cv2.circle(canvas, (x + 56, y + 13), 10, (0, 255, 0), 1, cv2.LINE_AA)

        # State text below housing
        badge_text = f"{state} {int(math.ceil(countdown))}s"
        cv2.putText(canvas, badge_text, (x - 2, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (0, 0, 255) if state == "RED" else ((0, 215, 255) if state == "YELLOW" else (0, 255, 0)), 1, cv2.LINE_AA)

    def update_and_render_vehicles(
        self,
        canvas: np.ndarray,
        frame_idx: int,
        tracked_vehicles: List[Dict[str, Any]],
        active_violations: Dict[int, List[str]]
    ) -> np.ndarray:
        """Mirror real-time vehicle positions, trajectory trails, and violation warning highlights."""
        out = canvas.copy()

        # Update trajectory history
        current_track_ids = set()
        for v in tracked_vehicles:
            tid = v["track_id"]
            current_track_ids.add(tid)
            # Contact point: bottom center of bounding box
            bx, by, bw, bh = v["bbox"]
            contact_cam = (bx + bw / 2.0, by + bh)
            tx, ty = self.project_point(contact_cam)

            if tid not in self.trajectories:
                self.trajectories[tid] = []
            self.trajectories[tid].append((tx, ty, frame_idx))
            if len(self.trajectories[tid]) > self.max_trail_len:
                self.trajectories[tid].pop(0)

        # Purge stale trajectories
        for tid in list(self.trajectories.keys()):
            if tid not in current_track_ids:
                if self.trajectories[tid]:
                    # Remove if last seen over 40 frames ago
                    if frame_idx - self.trajectories[tid][-1][2] > 40:
                        del self.trajectories[tid]

        # Draw trajectory trails with fading alpha
        for tid, pts in self.trajectories.items():
            if len(pts) < 2:
                continue
            is_violating = tid in active_violations and len(active_violations[tid]) > 0
            trail_base_color = (0, 50, 255) if is_violating else (0, 220, 200)

            for i in range(1, len(pts)):
                alpha = i / len(pts)
                color = (
                    int(trail_base_color[0] * alpha),
                    int(trail_base_color[1] * alpha),
                    int(trail_base_color[2] * alpha)
                )
                pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(out, pt1, pt2, color, thickness=max(1, int(3 * alpha)), lineType=cv2.LINE_AA)

        # Draw 2.5D vehicle avatars
        for v in tracked_vehicles:
            tid = v["track_id"]
            cls_name = v.get("class", "vehicle").lower()
            speed = v.get("speed_kmh", 0.0)
            is_violating = tid in active_violations and len(active_violations[tid]) > 0
            violations_list = active_violations.get(tid, [])

            # Contact point in twin
            bx, by, bw, bh = v["bbox"]
            contact_cam = (bx + bw / 2.0, by + bh)
            tx, ty = self.project_point(contact_cam)

            # Dimensions on twin canvas
            if "motorcycle" in cls_name or "bike" in cls_name:
                vw, vh = 16, 26
                avatar_color = (255, 180, 50)  # Cyan-blue
            elif "auto" in cls_name or "rickshaw" in cls_name:
                vw, vh = 24, 32
                avatar_color = (50, 220, 255)  # Yellow
            elif "bus" in cls_name or "truck" in cls_name:
                vw, vh = 36, 68
                avatar_color = (220, 100, 50)  # Blue
            else:  # car / van
                vw, vh = 28, 44
                avatar_color = (200, 200, 200)  # White/silver

            # If vehicle has active violation, override avatar color to intense red
            if is_violating:
                avatar_color = (0, 0, 255)
                # Pulsing warning halo
                pulse_rad = int(24 + 6 * math.sin(frame_idx * 0.4))
                cv2.circle(out, (tx, ty - vh // 2), pulse_rad, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Draw vehicle body (rounded rectangle)
            top_left = (tx - vw // 2, ty - vh)
            bottom_right = (tx + vw // 2, ty)
            cv2.rectangle(out, top_left, bottom_right, avatar_color, -1)
            cv2.rectangle(out, top_left, bottom_right, (255, 255, 255), 1)

            # Windshield / roof detail
            if vh > 20:
                cv2.rectangle(out, (tx - vw // 4, ty - int(vh * 0.75)), (tx + vw // 4, ty - int(vh * 0.35)), (40, 45, 55), -1)

            # Wheels / headlights
            cv2.circle(out, (tx - vw // 2 + 2, ty - vh + 3), 2, (0, 255, 255), -1)
            cv2.circle(out, (tx + vw // 2 - 2, ty - vh + 3), 2, (0, 255, 255), -1)

            # ID label & Speed
            label = f"#{tid}"
            cv2.putText(out, label, (tx - vw // 2, ty - vh - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

            # If violating, display violation badge above avatar
            if is_violating:
                v_badge = violations_list[0].replace("_", " ")
                cv2.rectangle(out, (tx - 50, ty - vh - 22), (tx + 50, ty - vh - 6), (0, 0, 180), -1)
                cv2.putText(out, v_badge[:14], (tx - 46, ty - vh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom Telemetry Overlay
        total_tracked = len(tracked_vehicles)
        total_violations = sum(len(v) for v in active_violations.values())
        cv2.rectangle(out, (0, self.canvas_h - 40), (self.canvas_w, self.canvas_h), (14, 18, 25), -1)
        telemetry_str = f"ACTIVE VEHICLES: {total_tracked:02d} | CUMULATIVE VIOLATIONS: {total_violations:02d}"
        cv2.putText(out, telemetry_str, (16, self.canvas_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 200), 1, cv2.LINE_AA)

        return out

    def composite_side_by_side(self, camera_frame: np.ndarray, digital_twin_frame: np.ndarray) -> np.ndarray:
        """Concatenate camera frame and digital twin side-by-side into a single canvas."""
        cam_h, cam_w = camera_frame.shape[:2]
        twin_h, twin_w = digital_twin_frame.shape[:2]

        # Standardize height
        target_h = max(cam_h, twin_h)
        if cam_h != target_h:
            camera_frame = cv2.resize(camera_frame, (int(cam_w * target_h / cam_h), target_h))
        if twin_h != target_h:
            digital_twin_frame = cv2.resize(digital_twin_frame, (int(twin_w * target_h / twin_h), target_h))

        # Horizontal stacking
        composite = np.hstack([camera_frame, digital_twin_frame])
        return composite
