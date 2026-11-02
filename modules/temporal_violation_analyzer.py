"""
Temporal association and violation confirmation.

Adds rider-to-motorcycle association, helmet vote history, triple-riding vote
history, and best-per-track plate caching without replacing the existing
tracking/lane modules.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .tracker_pipeline import TrackObservation


@dataclass
class PlateCandidate:
    plate_number: str = "UNKNOWN"
    confidence: float = 0.0
    frame: int = -1
    review_required: bool = True


@dataclass
class RiderState:
    rider_counts: deque = field(default_factory=lambda: deque(maxlen=8))
    no_helmet_votes: deque = field(default_factory=lambda: deque(maxlen=8))
    latest_helmet_conf: float = 0.0
    latest_rider_conf: float = 0.0
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)


class TemporalViolationAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        cfg = config.get("temporal_violations", {})
        self.rider_required_votes = int(cfg.get("rider_required_votes", 3))
        self.no_helmet_required_votes = int(cfg.get("no_helmet_required_votes", 3))
        self.association_iou_threshold = float(cfg.get("association_iou_threshold", 0.03))
        self.association_distance_px = float(cfg.get("association_distance_px", 180.0))
        self.plate_interval_frames = int(cfg.get("plate_interval_frames", 15))

        self.rider_states: Dict[int, RiderState] = defaultdict(RiderState)
        self.best_plates: Dict[int, PlateCandidate] = {}
        self.last_plate_frame: Dict[int, int] = defaultdict(lambda: -10**9)

    def update_rider_rules(
        self,
        vehicles: List[TrackObservation],
        pedestrians: List[TrackObservation],
        helmet_detections: List[Any],
        frame: Optional[np.ndarray] = None,
        helmet_model: Optional[Any] = None,
    ) -> Dict[int, Dict[str, Any]]:
        output: Dict[int, Dict[str, Any]] = {}
        for vehicle in vehicles:
            if vehicle.cls_name not in ("motorcycle", "bicycle"):
                continue

            state = self.rider_states[vehicle.track_id]
            state.detected_objects = [
                {
                    "class": vehicle.cls_name,
                    "confidence": round(float(vehicle.confidence), 3),
                    "bbox": [round(float(c), 1) for c in vehicle.bbox],
                    "role": "violator_vehicle",
                }
            ]

            # 1. Associated people from primary detector
            people = self._associated_people(vehicle, pedestrians)
            for p in people:
                state.detected_objects.append(
                    {
                        "class": p.cls_name,
                        "confidence": round(float(p.confidence), 3),
                        "bbox": [round(float(c), 1) for c in p.bbox],
                        "role": "associated_rider",
                    }
                )

            # 2. Targeted high-resolution ROI crop detection with helmet model
            active_helmets = list(helmet_detections)
            if frame is not None and helmet_model is not None:
                crop_dets = self._run_crop_helmet_inference(frame, vehicle, helmet_model)
                active_helmets.extend(crop_dets)

            # 3. Helmet & No-Helmet verification
            no_helmet, helmet_conf, helmet_obj = self._evaluate_helmet_presence(vehicle, people, active_helmets)
            if helmet_obj:
                state.detected_objects.append(helmet_obj)
            state.latest_helmet_conf = helmet_conf
            state.latest_rider_conf = max([p.confidence for p in people] or [vehicle.confidence])

            # Multi-cue rider count: max between detected people and detected head/helmet clusters
            head_count = sum(
                1 for h in active_helmets
                if self._point_in_box(
                    ((getattr(h, "bbox", [0, 0, 0, 0])[0] + getattr(h, "bbox", [0, 0, 0, 0])[2]) / 2.0,
                     (getattr(h, "bbox", [0, 0, 0, 0])[1] + getattr(h, "bbox", [0, 0, 0, 0])[3]) / 2.0),
                    (vehicle.bbox[0] - vehicle.width * 0.4,
                     vehicle.bbox[1] - vehicle.height * 0.8,
                     vehicle.bbox[2] + vehicle.width * 0.4,
                     vehicle.bbox[3])
                )
            )
            estimated_riders = max(len(people), head_count)

            state.rider_counts.append(estimated_riders)
            state.no_helmet_votes.append(bool(no_helmet))

            triple_votes = sum(1 for c in state.rider_counts if c >= 3)
            no_helmet_votes = sum(1 for v in state.no_helmet_votes if v)
            violations = []
            if triple_votes >= self.rider_required_votes:
                violations.append("TRIPLE_RIDING")
            if no_helmet_votes >= self.no_helmet_required_votes:
                violations.append("NO_HELMET")

            output[vehicle.track_id] = {
                "associated_person_track_ids": [p.track_id for p in people],
                "associated_person_count": estimated_riders,
                "rider_count_history": list(state.rider_counts),
                "no_helmet_vote_history": list(state.no_helmet_votes),
                "temporal_violations": violations,
                "confidence": {
                    "vehicle": round(float(vehicle.confidence), 3),
                    "helmet_model": round(float(state.latest_helmet_conf), 3),
                    "rider_detector": round(float(state.latest_rider_conf), 3),
                },
                "detected_objects": state.detected_objects,
            }
        return output

    def _run_crop_helmet_inference(
        self,
        frame: np.ndarray,
        vehicle: TrackObservation,
        helmet_model: Any,
    ) -> List[Any]:
        fh, fw = frame.shape[:2]
        vx1, vy1, vx2, vy2 = vehicle.bbox
        cfg = self.config.get("temporal_violations", {})
        pad_top = vehicle.height * float(cfg.get("two_wheeler_crop_padding_top", 0.55))
        pad_side = vehicle.width * float(cfg.get("two_wheeler_crop_padding_sides", 0.30))

        cy1 = max(0, int(vy1 - pad_top))
        cy2 = min(fh, int(vy2 + vehicle.height * 0.10))
        cx1 = max(0, int(vx1 - pad_side))
        cx2 = min(fw, int(vx2 + pad_side))

        if (cy2 - cy1) < 24 or (cx2 - cx1) < 24:
            return []

        crop = frame[cy1:cy2, cx1:cx2]
        device = self.config.get("secondary_models", {}).get("device", "cpu")
        try:
            results = helmet_model.predict(crop, conf=0.30, device=device, verbose=False)
            if not results or results[0].boxes is None:
                return []
            
            box_res = results[0].boxes
            boxes = box_res.xyxy.cpu().numpy()
            clss = box_res.cls.cpu().numpy().astype(int)
            confs = box_res.conf.cpu().numpy()

            extracted = []
            for b, c_idx, conf in zip(boxes, clss, confs):
                label = str(helmet_model.names.get(c_idx, f"cls_{c_idx}"))
                full_box = [float(b[0] + cx1), float(b[1] + cy1), float(b[2] + cx1), float(b[3] + cy1)]
                # Simple container matching SecondaryDetection interface
                extracted.append(
                    type("CropDet", (), {
                        "detector": "helmet",
                        "label": label,
                        "confidence": float(conf),
                        "bbox": full_box,
                    })()
                )
            return extracted
        except Exception:
            return []

    def _evaluate_helmet_presence(
        self,
        vehicle: TrackObservation,
        people: List[TrackObservation],
        helmet_detections: List[Any],
    ) -> Tuple[bool, float, Optional[Dict[str, Any]]]:
        if not helmet_detections:
            return False, 0.0, None

        regions = [p.bbox for p in people]
        if not regions:
            vx1, vy1, vx2, vy2 = vehicle.bbox
            regions = [(vx1 - vehicle.width * 0.4, vy1 - vehicle.height * 1.0, vx2 + vehicle.width * 0.4, vy2)]

        best_no_helmet_conf = 0.0
        best_det_obj = None

        for det in helmet_detections:
            label = str(getattr(det, "label", "")).lower()
            box = tuple(float(v) for v in getattr(det, "bbox", []))
            if len(box) != 4:
                continue
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            is_associated = any(
                self._point_in_box((cx, cy), region) or self._iou(box, region) > 0.01
                for region in regions
            )
            if not is_associated:
                continue

            conf = float(getattr(det, "confidence", 0.0))
            is_no_helmet = any(k in label for k in ["no_helmet", "no helmet", "without", "invalid"])

            if is_no_helmet and conf > best_no_helmet_conf:
                best_no_helmet_conf = conf
                best_det_obj = {
                    "class": "no_helmet",
                    "confidence": round(conf, 3),
                    "bbox": [round(c, 1) for c in box],
                    "role": "headgear_violation",
                }

        if best_no_helmet_conf > 0:
            return True, best_no_helmet_conf, best_det_obj

        # Check if verified helmets were detected
        for det in helmet_detections:
            label = str(getattr(det, "label", "")).lower()
            if "helmet" in label and not any(k in label for k in ["no", "without", "invalid"]):
                box = tuple(float(v) for v in getattr(det, "bbox", []))
                if len(box) == 4:
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    if any(self._point_in_box((cx, cy), r) for r in regions):
                        return False, float(getattr(det, "confidence", 0.0)), {
                            "class": "helmet",
                            "confidence": round(float(getattr(det, "confidence", 0.0)), 3),
                            "bbox": [round(c, 1) for c in box],
                            "role": "protective_headgear_compliant",
                        }

        return False, 0.0, None

    def update_plate(
        self,
        frame_idx: int,
        frame: np.ndarray,
        vehicle: TrackObservation,
        plate_reader: Any,
    ) -> PlateCandidate:
        if frame_idx - self.last_plate_frame[vehicle.track_id] < self.plate_interval_frames:
            return self.get_plate(vehicle.track_id)
        self.last_plate_frame[vehicle.track_id] = frame_idx

        try:
            result = plate_reader.read_plate(frame, vehicle_bbox=vehicle.bbox)
        except Exception:
            return self.get_plate(vehicle.track_id)

        candidate = PlateCandidate(
            plate_number=result.get("plate_number", "UNKNOWN"),
            confidence=float(result.get("confidence", 0.0) or 0.0),
            frame=frame_idx,
            review_required=bool(result.get("review_required", True)),
        )
        current = self.best_plates.get(vehicle.track_id)
        if current is None or candidate.confidence > current.confidence:
            self.best_plates[vehicle.track_id] = candidate
        return self.best_plates[vehicle.track_id]

    def get_plate(self, track_id: int) -> PlateCandidate:
        return self.best_plates.get(track_id, PlateCandidate())

    def _associated_people(
        self,
        vehicle: TrackObservation,
        pedestrians: List[TrackObservation],
    ) -> List[TrackObservation]:
        vx1, vy1, vx2, vy2 = vehicle.bbox
        vcx, vcy = vehicle.center
        expanded = (
            vx1 - vehicle.width * 0.65,
            vy1 - vehicle.height * 1.35,
            vx2 + vehicle.width * 0.65,
            vy2 + vehicle.height * 0.25,
        )
        people = []
        for person in pedestrians:
            pcx, pcy = person.center
            if self._point_in_box((pcx, pcy), expanded):
                people.append(person)
                continue
            if self._iou(expanded, person.bbox) >= self.association_iou_threshold:
                people.append(person)
                continue
            if np.hypot(pcx - vcx, pcy - vcy) <= self.association_distance_px and pcy <= vy2 + 30:
                people.append(person)
        return people

    def _point_in_box(self, point: Tuple[float, float], box: Tuple[float, float, float, float]) -> bool:
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def _iou(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
