#!/usr/bin/env python3
"""
calibrate_lanes.py

One-time manual lane calibration for a FIXED camera. Since the camera doesn't
move, you mark the real lane boundaries once by clicking along each painted
lane line (or its implied path) on a sample frame. This produces ground-truth
lane geometry -- accurate by construction, independent of traffic behavior.

For an N-lane road, you draw N+1 boundary lines (left edge of lane 1, between
lane 1/2, between lane 2/3, ..., right edge of last lane).

Controls:
    Left-click      Add a point to the CURRENT boundary line (click along the
                     lane marking from top of frame to bottom, or bottom to
                     top -- order doesn't matter, it gets sorted by y).
    'n'             Finish current boundary, start a new one.
    'u'             Undo last point on current boundary.
    'r'             Reset (clear) current boundary entirely.
    'z'             Delete the last COMPLETED boundary (in case you want to redo it).
    's'             Save all boundaries to --out and exit.
    'q' / ESC       Quit without saving.

Usage:
    python calibrate_lanes.py \
        --video path/to/video.mp4 \
        --frame_index 500 \
        --out lane_boundaries.json

Tip: click 4-8 points per boundary line, following any curvature caused by
camera perspective. More points = more accurate curve, especially useful
near the top of frame where perspective bends the lane most.
"""

import argparse
import json

import cv2
import numpy as np

CURRENT_COLOR = (0, 255, 255)   # yellow: boundary currently being drawn
DONE_COLOR = (0, 200, 0)        # green: completed boundaries
POINT_RADIUS = 4


class Calibrator:
    def __init__(self, frame):
        self.base_frame = frame
        self.completed_boundaries = []   # list of list[(x, y)]
        self.current_points = []         # list[(x, y)] for boundary in progress

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))

    def finish_current_boundary(self):
        if len(self.current_points) < 2:
            print("Need at least 2 points to complete a boundary. Ignored.")
            return
        # Sort by y so downstream interpolation (np.interp) works correctly
        pts = sorted(self.current_points, key=lambda p: p[1])
        self.completed_boundaries.append(pts)
        self.current_points = []
        print(f"Boundary {len(self.completed_boundaries)} completed "
              f"({len(pts)} points). Start clicking the next boundary "
              f"(or press 's' to save if this was the last one).")

    def undo_last_point(self):
        if self.current_points:
            self.current_points.pop()

    def reset_current(self):
        self.current_points = []

    def delete_last_completed(self):
        if self.completed_boundaries:
            removed = self.completed_boundaries.pop()
            print(f"Removed last completed boundary ({len(removed)} points).")

    def render(self):
        frame = self.base_frame.copy()

        for i, boundary in enumerate(self.completed_boundaries):
            pts = np.array(boundary, dtype=np.int32)
            for j in range(len(pts) - 1):
                cv2.line(frame, tuple(pts[j]), tuple(pts[j + 1]), DONE_COLOR, 2)
            for p in pts:
                cv2.circle(frame, tuple(p), POINT_RADIUS, DONE_COLOR, -1)
            label_pt = tuple(pts[len(pts) // 2])
            cv2.putText(frame, f"B{i}", (label_pt[0] + 8, label_pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, DONE_COLOR, 2)

        if self.current_points:
            pts = np.array(self.current_points, dtype=np.int32)
            for j in range(len(pts) - 1):
                cv2.line(frame, tuple(pts[j]), tuple(pts[j + 1]), CURRENT_COLOR, 2)
            for p in pts:
                cv2.circle(frame, tuple(p), POINT_RADIUS, CURRENT_COLOR, -1)

        n_lanes = max(len(self.completed_boundaries) - 1, 0)
        hud = [
            f"Completed boundaries: {len(self.completed_boundaries)}  "
            f"(-> {n_lanes} lane(s) so far)",
            f"Current boundary points: {len(self.current_points)}",
            "[click]=add point  n=finish boundary  u=undo point  "
            "r=reset current  z=delete last boundary  s=save  q=quit",
        ]
        for i, line in enumerate(hud):
            y = 25 + i * 22
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--frame_index", type=int, default=500, help="Frame to calibrate on")
    parser.add_argument("--out", required=True, help="Output JSON path for lane boundaries")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit(f"Could not read frame {args.frame_index}")

    h, w = frame.shape[:2]
    calib = Calibrator(frame)

    window = "Lane Calibration - see console for controls"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, calib.mouse_callback)

    print(__doc__)

    while True:
        cv2.imshow(window, calib.render())
        key = cv2.waitKey(20) & 0xFF

        if key == ord('n'):
            calib.finish_current_boundary()
        elif key == ord('u'):
            calib.undo_last_point()
        elif key == ord('r'):
            calib.reset_current()
        elif key == ord('z'):
            calib.delete_last_completed()
        elif key == ord('s'):
            if calib.current_points:
                print("You have an unfinished boundary in progress -- "
                      "press 'n' to complete it first, or 'r' to discard it.")
                continue
            if len(calib.completed_boundaries) < 2:
                print("Need at least 2 completed boundaries (to form >=1 lane) before saving.")
                continue
            data = {
                "frame_width": w,
                "frame_height": h,
                "boundaries": [
                    [[int(x), int(y)] for x, y in boundary]
                    for boundary in calib.completed_boundaries
                ],
            }
            with open(args.out, "w") as f:
                json.dump(data, f, indent=2)
            n_lanes = len(calib.completed_boundaries) - 1
            print(f"Saved {len(calib.completed_boundaries)} boundaries "
                  f"({n_lanes} lane(s)) to {args.out}")
            break
        elif key == ord('q') or key == 27:
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
