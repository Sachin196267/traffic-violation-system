"""
modules/tracker_pipeline.py

Object Detection and Multi-Object Tracking pipeline using YOLO + BoT-SORT.
Features:
- Extracts vehicle contact points (bottom-center) approximating road-tire contact.
- Separates pedestrians and cyclists/riders from vehicular traffic.
- Identifies parked and stationary vehicles to prevent lane distortion.
- Computes smoothed instantaneous velocities and overall movement vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from ultralytics import YOLO


@dataclass
class TrackObservation:
    track_id: int
    cls_name: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]             # (cx, cy)
    bottom_center: Tuple[float, float]      # (cx, y2) - road contact point
    width: float
    height: float
    confidence: float
    frame_idx: int


@dataclass
class TrackSummary:
    track_id: int
    cls_name: str
    observations: List[TrackObservation] = field(default_factory=list)
    is_parked: bool = False
    direction: str = "unknown"  # "inbound", "outbound", "lateral", "static"
    total_displacement: float = 0.0
    last_frame: int = 0

    @property
    def latest(self) -> Optional[TrackObservation]:
        return self.observations[-1] if self.observations else None

    @property
    def history_bottom_centers(self) -> List[Tuple[float, float]]:
        return [obs.bottom_center for obs in self.observations]

    @property
    def history_frames(self) -> List[int]:
        return [obs.frame_idx for obs in self.observations]

    def update_motion_stats(self, min_disp_px: float = 18.0):
        if len(self.observations) < 2:
            self.direction = "static"
            self.total_displacement = 0.0
            return

        p0 = np.array(self.observations[0].bottom_center)
        p_latest = np.array(self.observations[-1].bottom_center)
        diff = p_latest - p0
        self.total_displacement = float(np.linalg.norm(diff))

        if self.total_displacement < min_disp_px:
            self.is_parked = True
            self.direction = "static"
        else:
            self.is_parked = False
            dx, dy = diff[0], diff[1]
            if abs(dy) > abs(dx) * 0.4:
                self.direction = "inbound" if dy > 0 else "outbound"
            else:
                self.direction = "lateral"


class TrackerPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_cfg = config.get("model", {})
        tracking_cfg = config.get("tracking", {})
        classes_cfg = config.get("classes", {})

        model_path = model_cfg.get("path", "yolo11s.pt")
        self.conf_thresh = model_cfg.get("conf_threshold", 0.25)
        self.tracker_type = model_cfg.get("tracker", "botsort.yaml")
        self.device = model_cfg.get("device", "cpu")

        self.min_track_len = tracking_cfg.get("min_track_length", 8)
        self.min_disp_px = tracking_cfg.get("min_displacement_px", 18.0)
        self.contact_offset = tracking_cfg.get(
            "contact_point_offset",
            tracking_cfg.get("contact_offset", 0.98),
        )

        self.vehicle_classes = set(classes_cfg.get("vehicle", [
            "car", "bus", "truck", "motorcycle", "autorickshaw", "vehicle fallback"
        ]))
        self.pedestrian_classes = set(classes_cfg.get("pedestrian", ["person", "rider"]))

        print(f"[TrackerPipeline] Loading YOLO model: {model_path} on {self.device}...")
        self.model = YOLO(model_path)
        self.tracks: Dict[int, TrackSummary] = {}
        self.pedestrian_tracks: List[TrackObservation] = []

    def process_frame(
        self, frame: np.ndarray, frame_idx: int
    ) -> Tuple[List[TrackObservation], Dict[int, TrackSummary], List[TrackObservation]]:
        """
        Runs YOLO tracking on a single frame.
        Returns:
            - current_frame_vehicles: list of TrackObservation for active vehicles
            - all_vehicle_tracks: accumulated track summaries
            - current_frame_pedestrians: list of pedestrians detected in this frame
        """
        results = self.model.track(
            source=frame,
            tracker=self.tracker_type,
            conf=self.conf_thresh,
            persist=True,
            device=self.device,
            verbose=False,
        )

        current_vehicles: List[TrackObservation] = []
        current_pedestrians: List[TrackObservation] = []

        if not results or len(results) == 0:
            return current_vehicles, self.tracks, current_pedestrians

        res = results[0]
        if res.boxes is None or res.boxes.id is None:
            return current_vehicles, self.tracks, current_pedestrians

        boxes = res.boxes.xyxy.cpu().numpy()
        track_ids = res.boxes.id.cpu().numpy().astype(int)
        class_indices = res.boxes.cls.cpu().numpy().astype(int)
        confidences = res.boxes.conf.cpu().numpy()

        for box, tid, cls_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            cls_name = self.model.names.get(cls_idx, f"cls_{cls_idx}").lower()
            x1, y1, x2, y2 = box
            w = float(x2 - x1)
            h = float(y2 - y1)
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            
            # Bottom contact point calculation
            y_contact = float(y1 + h * self.contact_offset)
            contact_pt = (cx, y_contact)

            obs = TrackObservation(
                track_id=int(tid),
                cls_name=cls_name,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                center=(cx, cy),
                bottom_center=contact_pt,
                width=w,
                height=h,
                confidence=float(conf),
                frame_idx=frame_idx,
            )

            if cls_name in self.pedestrian_classes:
                current_pedestrians.append(obs)
                continue

            # Check if this class is treated as a vehicle
            is_vehicle = (
                cls_name in self.vehicle_classes or
                any(v in cls_name for v in ["car", "bus", "truck", "motorcycle", "auto", "vehicle"])
            )

            if is_vehicle:
                current_vehicles.append(obs)
                if tid not in self.tracks:
                    self.tracks[tid] = TrackSummary(track_id=int(tid), cls_name=cls_name)

                self.tracks[tid].observations.append(obs)
                self.tracks[tid].last_frame = frame_idx
                self.tracks[tid].update_motion_stats(min_disp_px=self.min_disp_px)

        return current_vehicles, self.tracks, current_pedestrians

    def get_clean_moving_tracks(
        self, direction: Optional[str] = None, min_obs: Optional[int] = None
    ) -> List[TrackSummary]:
        """
        Returns tracks that have moved sufficiently and are not parked.
        Can optionally filter by movement direction ('inbound' or 'outbound').
        """
        min_len = min_obs if min_obs is not None else self.min_track_len
        valid = []
        for track in self.tracks.values():
            if len(track.observations) < min_len:
                continue
            if track.is_parked or track.direction == "static":
                continue
            if direction and track.direction != direction:
                continue
            valid.append(track)
        return valid
