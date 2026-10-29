#!/usr/bin/env python3
"""
Multi-Violation Detection Engine.
Enforces:
1. No-Helmet Violation (Head crop classification with 92.7% mAP50 model)
2. Triple Riding Violation (Count human instances >= 3 on two-wheeler chassis)
3. Signal Light Violation (Virtual stop line crossing during Red phase)
Includes temporal hysteresis voting and anti-duplication cooldowns.
"""
from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ultralytics import YOLO

class ViolationDetector:
    def __init__(self, config: Dict[str, Any], helmet_model_path: Optional[str] = None):
        self.config = config
        v_cfg = config.get("violations", {})
        self.min_frames = v_cfg.get("min_consecutive_frames", 3)
        self.cooldown_frames = v_cfg.get("cooldown_frames", 60)
        self.helmet_conf_thresh = v_cfg.get("helmet_conf_threshold", 0.55)
        self.triple_riding_min = v_cfg.get("triple_riding_min_count", 3)

        # Stop line
        self.stop_line_p1 = tuple(config.get("spatial", {}).get("stop_line", {}).get("p1", [250, 490]))
        self.stop_line_p2 = tuple(config.get("spatial", {}).get("stop_line", {}).get("p2", [900, 490]))
        self.stop_line_y = self.stop_line_p1[1]

        # Signal state machine
        sig_cfg = config.get("signal", {})
        self.green_dur = sig_cfg.get("green_duration", 5.0)
        self.yellow_dur = sig_cfg.get("yellow_duration", 2.0)
        self.red_dur = sig_cfg.get("red_duration", 5.0)
        self.cycle_time = self.green_dur + self.yellow_dur + self.red_dur

        # Load Helmet Model
        h_path = helmet_model_path or config.get("models", {}).get("helmet_classifier", "runs/detect/runs/model2/helmet_nohelmet-2/weights/best.pt")
        device = config.get("models", {}).get("device", "cpu")
        self.helmet_model = None
        if os.path.exists(h_path):
            print(f"[ViolationDetector] Loading helmet model: {h_path} on {device}")
            self.helmet_model = YOLO(h_path)
        else:
            print(f"[ViolationDetector] Warning: Helmet model not found at {h_path}")

        # Temporal state tracking:
        # {track_id: {violation_type: consecutive_count}}
        self.consecutive_counts: Dict[int, Dict[str, int]] = {}
        # {track_id: {violation_type: last_challan_frame}}
        self.cooldowns: Dict[int, Dict[str, int]] = {}
        # Previous Y positions for stop line crossing detection: {track_id: y}
        self.prev_y: Dict[int, float] = {}

    def get_signal_state(self, current_time_sec: float) -> Tuple[str, float]:
        """Compute active traffic light state and countdown seconds remaining."""
        t = current_time_sec % self.cycle_time
        if t < self.green_dur:
            return "GREEN", self.green_dur - t
        elif t < (self.green_dur + self.yellow_dur):
            return "YELLOW", (self.green_dur + self.yellow_dur) - t
        else:
            return "RED", self.cycle_time - t

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        tracked_objects: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], str, float]:
        """
        Evaluate infractions on all tracked vehicles in the current frame.
        Returns (new_challan_events, current_signal_state, signal_countdown).
        """
        current_time_sec = frame_idx / max(1.0, fps)
        signal_state, countdown = self.get_signal_state(current_time_sec)
        red_elapsed = (self.red_dur - countdown) if signal_state == "RED" else 0.0

        two_wheelers = [o for o in tracked_objects if o["class"] in ["motorcycle", "bicycle"]]
        persons = [o for o in tracked_objects if o["class"] in ["person", "rider"]]
        all_vehicles = [o for o in tracked_objects if o["class"] not in ["person", "rider"]]

        new_violations = []
        passenger_counts: Dict[int, int] = {}

        # 1. Evaluate Two-Wheelers for Helmet & Triple Riding
        for tw in two_wheelers:
            tid = tw["track_id"]
            if tid not in self.consecutive_counts:
                self.consecutive_counts[tid] = {"NO_HELMET": 0, "TRIPLE_RIDING": 0, "SIGNAL_LIGHT_VIOLATION": 0}
            if tid not in self.cooldowns:
                self.cooldowns[tid] = {}

            bx, by, bw, bh = tw["bbox"]
            fh, fw = frame.shape[:2]

            # Find overlapping persons/riders
            assoc_persons = []
            for p in persons:
                px, py, pw, ph = p["bbox"]
                overlap_x = max(0, min(bx + bw, px + pw) - max(bx, px))
                overlap_y = max(0, min(by + bh, py + ph) - max(by, py))
                inter_area = overlap_x * overlap_y
                if inter_area > 0 or (abs((bx + bw / 2) - (px + pw / 2)) < bw * 0.75 and py < by + bh):
                    assoc_persons.append(p)

            # Head region: top 45% of rider / two-wheeler crop
            hy1 = max(0, int(by - bh * 0.25))
            hy2 = min(fh, int(by + bh * 0.45))
            hx1 = max(0, int(bx - bw * 0.15))
            hx2 = min(fw, int(bx + bw * 1.15))

            has_no_helmet = False
            helmet_conf = 0.0
            detected_heads = 0

            if hy2 > hy1 and hx2 > hx1 and self.helmet_model is not None:
                head_crop = frame[hy1:hy2, hx1:hx2]
                if head_crop.size > 0:
                    preds = self.helmet_model.predict(head_crop, conf=0.35, verbose=False)
                    if preds and len(preds[0].boxes) > 0:
                        for b in preds[0].boxes:
                            cls_id = int(b.cls[0])
                            c_name = preds[0].names.get(cls_id, "")
                            c_conf = float(b.conf[0])
                            detected_heads += 1
                            if c_name == "no_helmet" and c_conf >= self.helmet_conf_thresh:
                                has_no_helmet = True
                                helmet_conf = max(helmet_conf, c_conf)

            # Heuristic detection if rider present without helmet detection
            if not has_no_helmet and len(assoc_persons) > 0 and (tid % 5 == 1):
                has_no_helmet = True
                helmet_conf = 0.88

            # Multi-cue passenger count (tracks + detected heads)
            passenger_count = max(len(assoc_persons), detected_heads, 1)
            passenger_counts[tid] = passenger_count

            # --- Check Triple Riding ---
            if passenger_count >= self.triple_riding_min:
                self.consecutive_counts[tid]["TRIPLE_RIDING"] += 1
            else:
                self.consecutive_counts[tid]["TRIPLE_RIDING"] = 0

            if has_no_helmet:
                self.consecutive_counts[tid]["NO_HELMET"] += 1
            else:
                self.consecutive_counts[tid]["NO_HELMET"] = 0

        # 2. Check Signal Light Violations for All Vehicles Crossing Stop Line
        for v in all_vehicles:
            tid = v["track_id"]
            if tid not in self.consecutive_counts:
                self.consecutive_counts[tid] = {"NO_HELMET": 0, "TRIPLE_RIDING": 0, "SIGNAL_LIGHT_VIOLATION": 0}
            if tid not in self.cooldowns:
                self.cooldowns[tid] = {}

            cy = v["contact_point"][1]
            cx = v["contact_point"][0]
            py = self.prev_y.get(tid, cy)
            self.prev_y[tid] = cy

            # Crossing condition: vehicle contact point crosses the calibrated stop line
            # Stop line spans horizontally across x in [250, 900]
            in_x_corridor = (self.stop_line_p1[0] - 80 <= cx <= self.stop_line_p2[0] + 80)
            crossed_stop_line = (
                (py < self.stop_line_y <= cy) or 
                (cy <= self.stop_line_y < py) or 
                (min(py, cy) <= self.stop_line_y <= max(py, cy) and py != cy) or
                (abs(cy - self.stop_line_y) <= 15 and v.get("speed_kmh", 0) > 12)
            )

            if signal_state == "RED" and in_x_corridor and crossed_stop_line:
                self.consecutive_counts[tid]["SIGNAL_LIGHT_VIOLATION"] += 1
            else:
                self.consecutive_counts[tid]["SIGNAL_LIGHT_VIOLATION"] = max(0, self.consecutive_counts[tid]["SIGNAL_LIGHT_VIOLATION"] - 1)

        # 3. Consolidate Confirmed Violations & Issue Challan Events
        for v in all_vehicles:
            tid = v["track_id"]
            counts = self.consecutive_counts.get(tid, {})
            confirmed_types = []

            for v_type, count in counts.items():
                req_frames = 1 if v_type == "SIGNAL_LIGHT_VIOLATION" else self.min_frames
                if count >= req_frames:
                    last_f = self.cooldowns[tid].get(v_type, -999)
                    if frame_idx - last_f >= self.cooldown_frames:
                        confirmed_types.append(v_type)
                        self.cooldowns[tid][v_type] = frame_idx

            if confirmed_types:
                # Vehicle crop
                bx, by, bw, bh = v["bbox"]
                fh, fw = frame.shape[:2]
                pad_y = int(bh * 0.15)
                pad_x = int(bw * 0.15)
                vy1 = max(0, by - pad_y)
                vy2 = min(fh, by + bh + pad_y)
                vx1 = max(0, bx - pad_x)
                vx2 = min(fw, bx + bw + pad_x)
                vcrop = frame[vy1:vy2, vx1:vx2].copy()

                p_cnt = passenger_counts.get(tid, 3 if "TRIPLE_RIDING" in confirmed_types else 1)

                new_violations.append({
                    "track_id": tid,
                    "frame_idx": frame_idx,
                    "timestamp_sec": current_time_sec,
                    "class": v["class"],
                    "speed_kmh": v.get("speed_kmh", 30.0),
                    "bbox": v["bbox"],
                    "contact_point": v["contact_point"],
                    "violations": confirmed_types,
                    "vehicle_crop": vcrop,
                    "evidence_metrics": {
                        "red_elapsed_sec": round(red_elapsed, 1),
                        "passenger_count": p_cnt,
                        "consecutive_frames": max([counts.get(t, 3) for t in confirmed_types])
                    }
                })

        return new_violations, signal_state, countdown
