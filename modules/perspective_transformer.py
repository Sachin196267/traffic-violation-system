"""
modules/perspective_transformer.py

Inverse Perspective Mapping (IPM) and Bird's-Eye View (BEV) Transformation.
Features:
- Automatically estimates the road vanishing point from vehicle motion trajectory convergence.
- Computes perspective Homography H and inverse Homography H_inv.
- Maps vehicle road-contact points to BEV metric-like coordinates where lanes are parallel.
- Projects inferred BEV lane boundaries back to perspective image coordinates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2


class PerspectiveTransformer:
    def __init__(self, config: Dict[str, Any], frame_w: int, frame_h: int):
        self.config = config
        self.frame_w = frame_w
        self.frame_h = frame_h

        persp_cfg = config.get("perspective", {})
        self.ipm_w = persp_cfg.get("ipm_width", 600)
        self.ipm_h = persp_cfg.get("ipm_height", 900)
        self.auto_vp = persp_cfg.get("auto_vanishing_point", True)
        self.default_vp_y_ratio = persp_cfg.get("default_vp_y_ratio", 0.35)
        self.roi_top_ratio = persp_cfg.get("road_roi_top_ratio", 0.38)
        self.roi_bottom_ratio = persp_cfg.get("road_roi_bottom_ratio", 0.98)

        # Default initial vanishing point
        self.vanishing_point = (frame_w * 0.50, frame_h * self.default_vp_y_ratio)

        # Homography matrices
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None

        # Standard initial homography setup
        self._initialize_default_homography()

    def _initialize_default_homography(self):
        """Initializes a baseline homography based on default road trapezoid."""
        y_top = self.frame_h * self.roi_top_ratio
        y_bottom = self.frame_h * self.roi_bottom_ratio

        # Trapezoid in image space: narrower at top, wide at bottom
        src_pts = np.array([
            [self.frame_w * 0.15, y_bottom],   # bottom-left
            [self.frame_w * 0.85, y_bottom],   # bottom-right
            [self.frame_w * 0.65, y_top],      # top-right
            [self.frame_w * 0.35, y_top],      # top-left
        ], dtype=np.float32)

        dst_pts = np.array([
            [self.ipm_w * 0.10, self.ipm_h],
            [self.ipm_w * 0.90, self.ipm_h],
            [self.ipm_w * 0.90, 0],
            [self.ipm_w * 0.10, 0],
        ], dtype=np.float32)

        self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    def estimate_vanishing_point(
        self, track_lines: List[Tuple[float, float, float, float]]
    ) -> Tuple[float, float]:
        """
        Estimates the vanishing point by finding the consensus intersection of moving track lines.
        track_lines: list of (x_start, y_start, x_end, y_end)
        """
        if len(track_lines) < 4:
            return self.vanishing_point

        lines_abc = []
        for x1, y1, x2, y2 in track_lines:
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            if length < 25.0:
                continue
            # Line in form a*x + b*y + c = 0
            # dy*x - dx*y + (dx*y1 - dy*x1) = 0
            a = dy
            b = -dx
            c = dx * y1 - dy * x1
            norm = np.hypot(a, b)
            if norm > 1e-5:
                lines_abc.append((a / norm, b / norm, c / norm))

        if len(lines_abc) < 4:
            return self.vanishing_point

        # RANSAC intersection of line pairs
        best_vp = self.vanishing_point
        max_inliers = 0
        n_lines = len(lines_abc)

        for _ in range(120):
            idx1, idx2 = np.random.choice(n_lines, 2, replace=False)
            a1, b1, c1 = lines_abc[idx1]
            a2, b2, c2 = lines_abc[idx2]

            det = a1 * b2 - a2 * b1
            if abs(det) < 0.05:
                continue  # Parallel lines, no clear intersection

            x_int = (b1 * c2 - b2 * c1) / det
            y_int = (c1 * a2 - c2 * a1) / det

            # Vanishing point should be reasonably near upper portion of frame
            if not (-self.frame_w * 0.5 <= x_int <= self.frame_w * 1.5):
                continue
            if not (0 <= y_int <= self.frame_h * 0.65):
                continue

            # Count inlier lines that pass close to this intersection
            dists = [
                abs(a * x_int + b * y_int + c) for a, b, c in lines_abc
            ]
            inliers = sum(d < 15.0 for d in dists)
            if inliers > max_inliers:
                max_inliers = inliers
                best_vp = (float(x_int), float(y_int))

        self.vanishing_point = best_vp
        return best_vp

    def update_homography_from_corridor(
        self, x_left_bot: float, x_right_bot: float, x_left_top: float, x_right_top: float
    ):
        """
        Updates the homography matrix using the detected carriageway boundaries.
        """
        y_top = max(10.0, self.frame_h * self.roi_top_ratio)
        y_bot = self.frame_h * self.roi_bottom_ratio

        src_pts = np.array([
            [x_left_bot, y_bot],
            [x_right_bot, y_bot],
            [x_right_top, y_top],
            [x_left_top, y_top],
        ], dtype=np.float32)

        # Map directly to bird's-eye canvas width/height
        dst_pts = np.array([
            [0.0, self.ipm_h],
            [float(self.ipm_w), self.ipm_h],
            [float(self.ipm_w), 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)

        try:
            self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self.H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        except Exception as e:
            print(f"[PerspectiveTransformer] Failed to update homography: {e}")

    def image_to_bev(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms (N, 2) image coordinates to (N, 2) bird's-eye view coordinates.
        """
        if self.H is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, self.H)
        return warped.reshape(-1, 2)

    def bev_to_image(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms (N, 2) bird's-eye view coordinates back to (N, 2) image coordinates.
        """
        if self.H_inv is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        unwarped = cv2.perspectiveTransform(pts, self.H_inv)
        return unwarped.reshape(-1, 2)

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        """Warps an image frame into the top-down bird's-eye representation."""
        if self.H is None:
            return np.zeros((self.ipm_h, self.ipm_w, 3), dtype=np.uint8)
        return cv2.warpPerspective(frame, self.H, (self.ipm_w, self.ipm_h))
