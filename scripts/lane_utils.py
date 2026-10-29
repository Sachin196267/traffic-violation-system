"""
lane_utils.py

Loads manually-calibrated lane boundaries (from calibrate_lanes.py) and
provides a fast lookup: given a vehicle's (x, y) position, which lane is it in?

This replaces trajectory-based lane inference with ground-truth geometry, so
lane assignment accuracy no longer depends on how vehicles happened to drive.
"""

import json

import numpy as np


class LaneMap:
    def __init__(self, boundaries_json_path):
        with open(boundaries_json_path) as f:
            data = json.load(f)

        self.frame_width = data["frame_width"]
        self.frame_height = data["frame_height"]

        # Each boundary: array of (y, x) sorted by y, ready for np.interp
        self.boundaries = []
        for boundary in data["boundaries"]:
            pts = np.array(boundary, dtype=float)  # columns: x, y
            order = np.argsort(pts[:, 1])
            ys = pts[order, 1]
            xs = pts[order, 0]
            self.boundaries.append((ys, xs))

        self.n_lanes = len(self.boundaries) - 1
        if self.n_lanes < 1:
            raise ValueError("Need at least 2 boundaries to define >=1 lane.")

    def boundary_x_at(self, boundary_idx, y):
        """Interpolated (or clamped-extrapolated) x of a boundary at height y."""
        ys, xs = self.boundaries[boundary_idx]
        # np.interp clamps to the endpoint values outside the observed range,
        # which is a reasonable default since lane boundaries are roughly
        # straight/smoothly-curved and unlikely to bend sharply beyond
        # calibrated points.
        return float(np.interp(y, ys, xs))

    def lane_index_at(self, x, y):
        """
        Return the lane index (0 = leftmost) containing point (x, y),
        or None if the point falls outside all calibrated lanes.
        """
        boundary_xs = [self.boundary_x_at(i, y) for i in range(len(self.boundaries))]

        if x < boundary_xs[0] or x > boundary_xs[-1]:
            return None

        for lane_idx in range(self.n_lanes):
            if boundary_xs[lane_idx] <= x <= boundary_xs[lane_idx + 1]:
                return lane_idx
        return None

    def lane_polygon(self, lane_idx, n_samples=40):
        """
        Return a polygon (Nx2 int array) for lane `lane_idx`, useful for
        cv2.pointPolygonTest, drawing, or masking. Built by sampling both
        of the lane's boundaries over their shared y-range.
        """
        left_ys, left_xs = self.boundaries[lane_idx]
        right_ys, right_xs = self.boundaries[lane_idx + 1]

        y_lo = max(left_ys.min(), right_ys.min())
        y_hi = min(left_ys.max(), right_ys.max())
        ys = np.linspace(y_lo, y_hi, n_samples)

        left_pts = [(self.boundary_x_at(lane_idx, y), y) for y in ys]
        right_pts = [(self.boundary_x_at(lane_idx + 1, y), y) for y in ys[::-1]]

        polygon = np.array(left_pts + right_pts, dtype=np.int32)
        return polygon


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python lane_utils.py lane_boundaries.json")
        sys.exit(1)
    lm = LaneMap(sys.argv[1])
    print(f"Loaded {lm.n_lanes} lane(s) from {sys.argv[1]}")
    test_y = lm.frame_height * 0.7
    for lane in range(lm.n_lanes):
        poly = lm.lane_polygon(lane)
        print(f"  Lane {lane}: polygon with {len(poly)} points, "
              f"x-range at y={test_y:.0f}: "
              f"{lm.boundary_x_at(lane, test_y):.0f} to {lm.boundary_x_at(lane + 1, test_y):.0f}")
