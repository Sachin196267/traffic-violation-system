#!/usr/bin/env python3
"""
ROI Calibration & Visualizer Tool.
Draws the imaginary lane polygon and stop line over frame 0 of raw_input.mp4.
"""
import os
import cv2
import yaml
import numpy as np

def calibrate(config_path="traffic_violation/final_video_testing/config.yaml", output_path="traffic_violation/final_video_testing/roi_preview.jpg"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    video_path = cfg["video"]["input_path"]
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read from {video_path}")

    h, w, _ = frame.shape
    overlay = frame.copy()

    # Draw imaginary virtual lane polygon (semi-transparent cyan fill)
    pts = np.array(cfg["spatial"]["virtual_lane_polygon"], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (255, 230, 0))  # BGR Cyan-yellow
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    # Draw stop line (thick red line)
    p1 = tuple(cfg["spatial"]["stop_line"]["p1"])
    p2 = tuple(cfg["spatial"]["stop_line"]["p2"])
    cv2.line(frame, p1, p2, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)

    # Text annotations
    cv2.putText(frame, "IMAGINARY VIRTUAL LANE CORRIDOR", (pts[0][0] - 80, pts[0][1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "VIRTUAL STOP LINE", (p1[0], p1[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, frame)
    print(f"[Calibrate] Saved ROI calibration preview to {output_path}")

if __name__ == "__main__":
    calibrate()
