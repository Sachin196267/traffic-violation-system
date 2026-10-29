#!/usr/bin/env python3
"""
select_roi.py

One-time click: draw a rough polygon around ONLY the carriageway/road area
you want lane detection to run on (exclude sidewalks, bus stops, medians,
and any opposite-direction traffic). This is NOT per-lane -- just one
polygon covering the whole road region of interest. Since the camera is
fixed, you do this once.

Controls:
    Left-click      Add a polygon point (go around the road's outline in order --
                     e.g. top-left corner of the road, top-right, bottom-right,
                     bottom-left, back to start).
    'u'             Undo last point.
    'r'             Reset (clear all points).
    's'             Save polygon to --out and exit (needs >= 3 points).
    'q' / ESC       Quit without saving.

Usage:
    python select_roi.py \
        --video path/to/video.mp4 \
        --frame_index 500 \
        --out road_roi.json
"""

import argparse
import json

import cv2
import numpy as np

POINT_COLOR = (0, 255, 255)
LINE_COLOR = (0, 200, 0)
FILL_COLOR = (0, 200, 0)


class RoiSelector:
    def __init__(self, frame):
        self.base_frame = frame
        self.points = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))

    def undo(self):
        if self.points:
            self.points.pop()

    def reset(self):
        self.points = []

    def render(self):
        frame = self.base_frame.copy()

        if len(self.points) >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(self.points, dtype=np.int32)], FILL_COLOR)
            frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        pts = self.points
        for i in range(len(pts)):
            cv2.circle(frame, pts[i], 5, POINT_COLOR, -1)
            if i > 0:
                cv2.line(frame, pts[i - 1], pts[i], LINE_COLOR, 2)
        if len(pts) >= 3:
            cv2.line(frame, pts[-1], pts[0], LINE_COLOR, 1)  # preview closing edge

        hud = [
            f"Points: {len(pts)}  (need >= 3 to save)",
            "[click]=add point  u=undo  r=reset  s=save  q=quit",
            "Trace ONLY the road area you want lanes detected on "
            "(exclude sidewalks, opposite lanes, medians).",
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
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame_index", type=int, default=500)
    parser.add_argument("--out", required=True)
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
    sel = RoiSelector(frame)

    window = "Select Road ROI - see console for controls"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, sel.mouse_callback)
    print(__doc__)

    while True:
        cv2.imshow(window, sel.render())
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u'):
            sel.undo()
        elif key == ord('r'):
            sel.reset()
        elif key == ord('s'):
            if len(sel.points) < 3:
                print("Need at least 3 points before saving.")
                continue
            data = {
                "frame_width": w,
                "frame_height": h,
                "polygon": [[int(x), int(y)] for x, y in sel.points],
            }
            with open(args.out, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved ROI polygon ({len(sel.points)} points) to {args.out}")
            break
        elif key == ord('q') or key == 27:
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
