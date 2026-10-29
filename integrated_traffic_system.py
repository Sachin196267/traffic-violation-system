#!/usr/bin/env python3
"""
Integrated traffic violation system.

One entry point for:
  - traffic object detection and tracking
  - automatic lane inference from vehicle trajectories
  - lane assignment and movement violations
  - license plate and emergency vehicle detector integration
  - optional helmet detector integration when a helmet model is configured
  - annotated video plus JSON/CSV reports
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO

from modules import (
    LaneAssigner,
    LaneVisualizer,
    LicensePlateReader,
    PerspectiveTransformer,
    SceneAnalyzer,
    SeverityEngine,
    SmartSignalController,
    TemporalLaneModel,
    TemporalViolationAnalyzer,
    TrackerPipeline,
    TrajectoryClusterer,
    ViolationSlipGenerator,
    ViolationXAIExplainer,
)


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class SecondaryDetection:
    frame: int
    detector: str
    label: str
    confidence: float
    bbox: List[float]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_file(path: Optional[str], label: str, required: bool = True) -> bool:
    if not path:
        if required:
            print(f"[WARN] {label} path is not configured.")
        return False
    if not os.path.exists(path):
        msg = f"{label} file not found: {path}"
        if required:
            sys.exit(msg)
        print(f"[WARN] {msg}")
        return False
    return True


def load_secondary_models(config: Dict[str, Any]) -> Dict[str, YOLO]:
    secondary_cfg = config.get("secondary_models", {})
    if not secondary_cfg.get("enabled", True):
        return {}

    models: Dict[str, YOLO] = {}
    for detector_name in ("license_plate", "emergency_vehicle", "helmet"):
        detector_cfg = secondary_cfg.get(detector_name, {})
        if not detector_cfg.get("enabled", False):
            continue
        model_path = detector_cfg.get("path")
        if ensure_file(model_path, detector_name, required=False):
            print(f"[IntegratedSystem] Loading {detector_name}: {model_path}")
            models[detector_name] = YOLO(model_path)
    return models


def run_secondary_detectors(
    frame: np.ndarray,
    frame_idx: int,
    config: Dict[str, Any],
    models: Dict[str, YOLO],
) -> List[SecondaryDetection]:
    secondary_cfg = config.get("secondary_models", {})
    device = secondary_cfg.get("device", config.get("model", {}).get("device", "cpu"))
    detections: List[SecondaryDetection] = []

    for detector_name, model in models.items():
        detector_cfg = secondary_cfg.get(detector_name, {})
        conf = float(detector_cfg.get("conf_threshold", 0.35))
        results = model.predict(frame, conf=conf, device=device, verbose=False)
        if not results:
            continue
        result = results[0]
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_idx = int(box.cls)
            label = str(model.names.get(cls_idx, f"cls_{cls_idx}"))
            xyxy = [round(float(v), 2) for v in box.xyxy.cpu().numpy()[0].tolist()]
            detections.append(
                SecondaryDetection(
                    frame=frame_idx,
                    detector=detector_name,
                    label=label,
                    confidence=round(float(box.conf), 3),
                    bbox=xyxy,
                )
            )
    return detections


def draw_secondary_detections(frame: np.ndarray, detections: List[SecondaryDetection]) -> np.ndarray:
    colors = {
        "license_plate": (0, 255, 255),
        "emergency_vehicle": (0, 120, 255),
        "helmet": (0, 0, 255),
    }
    out = frame
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = colors.get(det.detector, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.detector}:{det.label} {det.confidence:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyJSONEncoder)


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated Indian traffic violation system")
    parser.add_argument("--video", required=True, help="Input traffic video")
    parser.add_argument("--config", default="config/integrated_system_config.yaml")
    parser.add_argument("--output_dir", default="outputs/integrated_run")
    parser.add_argument("--max_frames", type=int, default=500, help="-1 processes the full video")
    parser.add_argument("--warmup_frames", type=int, default=40)
    parser.add_argument("--update_interval", type=int, default=10)
    parser.add_argument("--save_birdseye_video", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"Video file not found: {args.video}")
    if not os.path.exists(args.config):
        sys.exit(f"Config file not found: {args.config}")

    os.makedirs(args.output_dir, exist_ok=True)
    snapshots_dir = os.path.join(args.output_dir, "snapshots")
    slips_dir = os.path.join(args.output_dir, "violation_slips")
    os.makedirs(snapshots_dir, exist_ok=True)
    os.makedirs(slips_dir, exist_ok=True)

    config = load_config(args.config)
    ensure_file(config.get("model", {}).get("path"), "traffic model", required=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Unable to open video: {args.video}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = total_frames if args.max_frames <= 0 else min(args.max_frames, total_frames)

    print("=" * 72)
    print("INTEGRATED TRAFFIC VIOLATION SYSTEM")
    print(f"Video: {args.video}")
    print(f"Resolution: {frame_w}x{frame_h} @ {fps:.1f} fps")
    print(f"Frames: {n_frames} / {total_frames}")
    print(f"Output: {args.output_dir}")
    print("=" * 72)

    tracker = TrackerPipeline(config)
    transformer = PerspectiveTransformer(config, frame_w, frame_h)
    scene_analyzer = SceneAnalyzer(config, frame_w, frame_h)
    clusterer = TrajectoryClusterer(config)
    temporal_model = TemporalLaneModel(config)
    lane_assigner = LaneAssigner(config)
    visualizer = LaneVisualizer(config, frame_w, frame_h)
    signal_controller = SmartSignalController(config, frame_w, frame_h)
    temporal_violation_analyzer = TemporalViolationAnalyzer(config)
    severity_engine = SeverityEngine(config)
    xai_explainer = ViolationXAIExplainer()
    slip_generator = ViolationSlipGenerator(slips_dir, config=config)
    secondary_models = load_secondary_models(config)
    plate_model_path = config.get("secondary_models", {}).get("license_plate", {}).get("path", "models/license_plate_best.pt")
    anpr_reader = LicensePlateReader(model_path=plate_model_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    annotated_path = os.path.join(args.output_dir, "integrated_annotated.mp4")
    writer = cv2.VideoWriter(annotated_path, fourcc, fps, (frame_w, frame_h))

    bev_writer = None
    bev_path = os.path.join(args.output_dir, "integrated_birdseye.mp4")
    if args.save_birdseye_video:
        bev_writer = cv2.VideoWriter(bev_path, fourcc, fps, (transformer.ipm_w, transformer.ipm_h))

    all_assignments: List[Dict[str, Any]] = []
    all_secondary: List[SecondaryDetection] = []
    signal_decisions: List[Dict[str, Any]] = []
    violation_rows: List[Dict[str, Any]] = []
    issued_challans: Dict[Tuple[int, str], int] = {}
    cooldown_frames = int(config.get("reporting", {}).get("violation_cooldown_frames", 300))
    run_every = max(1, int(config.get("secondary_models", {}).get("run_every_n_frames", 5)))
    save_snapshots = bool(config.get("reporting", {}).get("save_violation_snapshots", True))

    start = time.time()
    pbar = tqdm(total=n_frames, desc="Processing", unit="frame")
    last_annotated = None

    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        vehicles, track_summaries, pedestrians = tracker.process_frame(frame, frame_idx)
        scene_analyzer.accumulate_occupancy(vehicles, pedestrians)
        clean_tracks = tracker.get_clean_moving_tracks()

        if frame_idx >= args.warmup_frames and (
            frame_idx % args.update_interval == 0 or not temporal_model.is_initialized
        ):
            track_lines = []
            for track in clean_tracks:
                pts = track.history_bottom_centers
                if len(pts) >= 2:
                    track_lines.append((pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]))

            vp = transformer.estimate_vanishing_point(track_lines)
            corridors, dividers = scene_analyzer.analyze_corridors_and_dividers(
                clean_tracks,
                vp,
                transformer=transformer,
            )

            new_lanes = []
            for corridor in corridors:
                corridor_transformer = PerspectiveTransformer(config, frame_w, frame_h)
                corridor_transformer.update_homography_from_corridor(
                    corridor.x_left_bottom,
                    corridor.x_right_bottom,
                    corridor.x_left_top,
                    corridor.x_right_top,
                )
                new_lanes.extend(
                    clusterer.infer_lanes_for_corridor(corridor, clean_tracks, corridor_transformer)
                )
            for lane_idx, lane in enumerate(new_lanes):
                lane.lane_id = lane_idx
            temporal_model.update(new_lanes, frame_idx)

        active_lanes = temporal_model.active_lanes
        secondary = []
        if secondary_models and frame_idx % run_every == 0:
            secondary = run_secondary_detectors(frame, frame_idx, config, secondary_models)
            all_secondary.extend(secondary)

        helmet_detections = [d for d in secondary if d.detector == "helmet"]
        rider_temporal = temporal_violation_analyzer.update_rider_rules(
            vehicles=vehicles,
            pedestrians=pedestrians,
            helmet_detections=helmet_detections,
            frame=frame,
            helmet_model=secondary_models.get("helmet"),
        )

        current_phase = signal_controller.current_decision.active_phase if signal_controller.current_decision else None

        assignments = lane_assigner.assign_vehicles(
            vehicles,
            track_summaries,
            active_lanes,
            corridors=scene_analyzer.corridors,
            frame_shape=(frame_h, frame_w),
            transformer=transformer,
            fps=fps,
            signal_phase=current_phase,
            secondary_detections=secondary,
        )

        # Suppress violations during warmup or when temporal lane model has not initialized
        if frame_idx < args.warmup_frames or not temporal_model.is_initialized:
            for a in assignments:
                a.is_violating = False
                a.violation_types = []

        vehicle_obs_by_id = {v.track_id: v for v in vehicles}
        for assignment in assignments:
            temporal_info = rider_temporal.get(assignment.track_id, {})
            for violation in temporal_info.get("temporal_violations", []):
                if violation not in assignment.violation_types:
                    assignment.violation_types.append(violation)
                    assignment.is_violating = True

        signal_decision = signal_controller.update(
            frame_idx=frame_idx,
            assignments=assignments,
            secondary_detections=secondary,
            lane_confidence=temporal_model.global_confidence,
        )
        signal_record = {
            "frame": frame_idx,
            "reason": signal_decision.reason,
            "active_phase": asdict(signal_decision.active_phase),
            "approach_state": signal_decision.approach_state,
            "top_scenarios": [
                asdict(s)
                for s in sorted(signal_decision.scenarios, key=lambda s: s.estimated_wait_seconds)[:5]
            ],
        }
        signal_decisions.append(signal_record)

        frame_record = {
            "frame": frame_idx,
            "num_active_lanes": len(active_lanes),
            "lane_confidence": round(temporal_model.global_confidence, 3),
            "vehicles": [
                {
                    "track_id": a.track_id,
                    "class": a.cls_name,
                    "lane_id": a.lane_id,
                    "lane_name": a.lane_name,
                    "lane_position_status": a.lane_position_status,
                    "confidence": a.lane_conf,
                    "direction": a.direction,
                    "speed_kmh": a.speed_kmh,
                    "rider_association": rider_temporal.get(a.track_id, {}),
                    "contact_point": [round(c, 1) for c in a.contact_point],
                    "is_violating": a.is_violating,
                    "violations": a.violation_types,
                }
                for a in assignments
            ],
            "secondary_detections": [asdict(d) for d in secondary],
            "signal_decision": signal_record,
        }

        for assignment in assignments:
            if not assignment.is_violating:
                continue

            # Check per-track debouncing/cooldown to prevent duplicate ticket spam
            violation_sig = "|".join(sorted(assignment.violation_types))
            last_issued = issued_challans.get((assignment.track_id, violation_sig), -999999)
            if (frame_idx - last_issued) < cooldown_frames:
                continue
            issued_challans[(assignment.track_id, violation_sig)] = frame_idx

            trk = track_summaries.get(assignment.track_id)
            bbox = trk.latest.bbox if (trk and trk.latest) else None
            trajectory = trk.history_bottom_centers[-30:] if trk else None

            snapshot_path = ""
            if save_snapshots:
                snapshot_path = os.path.join(
                    snapshots_dir,
                    f"frame_{frame_idx:06d}_track_{assignment.track_id}.jpg",
                )

            vehicle_obs = vehicle_obs_by_id.get(assignment.track_id)
            if vehicle_obs is not None:
                plate_candidate = temporal_violation_analyzer.update_plate(
                    frame_idx=frame_idx,
                    frame=frame,
                    vehicle=vehicle_obs,
                    plate_reader=anpr_reader,
                )
            else:
                plate_candidate = temporal_violation_analyzer.get_plate(assignment.track_id)
            plate_number = plate_candidate.plate_number
            plate_confidence = plate_candidate.confidence
            plate_review_required = plate_candidate.review_required

            temporal_info = rider_temporal.get(assignment.track_id, {})
            det_objs = temporal_info.get("detected_objects", [])
            det_conf = temporal_info.get("confidence", {}).get("helmet_model")
            if not det_conf or det_conf <= 0:
                det_conf = temporal_info.get("confidence", {}).get("vehicle", 0.88)

            severity = severity_engine.evaluate(
                violation_types=assignment.violation_types,
                cls_name=assignment.cls_name,
                speed_kmh=assignment.speed_kmh,
                lane_confidence=temporal_model.global_confidence,
            )

            xai = xai_explainer.explain(
                assignment=assignment,
                lane_confidence=temporal_model.global_confidence,
                signal_reason=signal_decision.reason,
                emergency_priority=signal_decision.active_phase.emergency_priority,
                detected_objects=det_objs,
                detection_confidence=det_conf,
                track_summary=trk,
                secondary_detections=secondary,
                bbox=bbox,
                timestamp_str=f"{frame_idx / fps:.2f}s" if fps > 0 else f"Frame {frame_idx}",
            )
            slip = slip_generator.generate(
                assignment=assignment,
                frame_idx=frame_idx,
                snapshot_path=snapshot_path,
                xai=xai,
                signal_record=signal_record,
                secondary_detections=secondary,
                video_name=args.video,
                frame=frame,
                bbox=bbox,
                trajectory=trajectory,
                plate_number=plate_number,
                plate_confidence=plate_confidence,
                plate_review_required=plate_review_required,
                severity=severity,
                detection_confidence=det_conf,
            )
            for vehicle_record in frame_record["vehicles"]:
                if vehicle_record["track_id"] == assignment.track_id:
                    vehicle_record["xai_explanation"] = xai.to_dict()
                    vehicle_record["plate_number"] = plate_number
                    vehicle_record["speed_kmh"] = assignment.speed_kmh
                    vehicle_record["severity"] = severity.to_dict()
                    vehicle_record["violation_slip"] = {
                        "path": slip.path,
                        "format": slip.format,
                        "html_path": slip.html_path,
                        "pdf_path": slip.pdf_path,
                        "amount": slip.amount,
                        "challan_id": slip.challan_id,
                        "plate_number": plate_number,
                        "plate_confidence": plate_confidence,
                        "severity_level": slip.severity_level,
                        "severity_score": slip.severity_score,
                        "detection_confidence": slip.detection_confidence,
                        "evidence_path": slip.evidence_path,
                        "evidence_valid": slip.evidence_valid,
                        "review_required": slip.review_required,
                        "rider_association": temporal_info,
                    }
            violation_rows.append(
                {
                    "frame": frame_idx,
                    "track_id": assignment.track_id,
                    "class": assignment.cls_name,
                    "plate_number": plate_number,
                    "lane_id": assignment.lane_id,
                    "lane_name": assignment.lane_name,
                    "direction": assignment.direction,
                    "speed_kmh": assignment.speed_kmh,
                    "violations": "|".join(assignment.violation_types),
                    "severity_level": slip.severity_level,
                    "severity_score": slip.severity_score,
                    "severity_reasons": "; ".join(severity.reasons),
                    "detection_confidence": slip.detection_confidence,
                    "snapshot_path": snapshot_path,
                    "challan_id": slip.challan_id,
                    "xai_justification": xai.primary_justification,
                    "slip_path": slip.path,
                    "slip_format": slip.format,
                    "html_receipt": slip.html_path,
                    "pdf_receipt": slip.pdf_path,
                    "evidence_path": slip.evidence_path,
                    "evidence_valid": slip.evidence_valid,
                    "review_required": slip.review_required,
                    "plate_confidence": plate_confidence,
                    "xai_score": xai.final_score,
                    "xai_top_reason": xai.top_reasons[0]["feature"] if xai.top_reasons else "",
                }
            )
        all_assignments.append(frame_record)

        annotated = visualizer.render_frame(
            frame=frame,
            active_lanes=active_lanes,
            dividers=scene_analyzer.dividers,
            assignments=assignments,
            current_vehicles=vehicles,
            track_summaries=track_summaries,
            transformer=transformer,
            frame_idx=frame_idx,
            fps=fps,
        )
        annotated = draw_secondary_detections(annotated, secondary)
        annotated = signal_controller.render_overlay(annotated, signal_decision)
        writer.write(annotated)
        last_annotated = annotated

        if bev_writer is not None:
            bev_writer.write(
                visualizer.render_birdseye_canvas(
                    active_lanes,
                    scene_analyzer.dividers,
                    track_summaries,
                    transformer,
                )
            )

        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    if bev_writer is not None:
        bev_writer.release()

    lane_boundaries = temporal_model.export_boundaries_json(frame_w, frame_h)
    lane_boundaries_path = os.path.join(args.output_dir, "lane_boundaries.json")
    write_json(lane_boundaries_path, lane_boundaries)

    tracks_json = [
        {
            "track_id": track.track_id,
            "class": track.cls_name,
            "direction": track.direction,
            "is_parked": track.is_parked,
            "total_displacement_px": round(track.total_displacement, 2),
            "num_observations": len(track.observations),
            "last_frame": track.last_frame,
            "latest_bbox": list(track.latest.bbox) if track.latest else None,
            "latest_contact_point": list(track.latest.bottom_center) if track.latest else None,
            "trajectory_contact_points": [
                [round(float(x), 1), round(float(y), 1)]
                for x, y in track.history_bottom_centers
            ],
        }
        for track in track_summaries.values()
    ]

    lanes_json = {
        "frame_width": frame_w,
        "frame_height": frame_h,
        "num_lanes": len(temporal_model.active_lanes),
        "confidence": round(temporal_model.global_confidence, 3),
        "vanishing_point": [round(c, 1) for c in transformer.vanishing_point],
        "boundaries": lane_boundaries,
        "lanes": [
            {
                "lane_id": lane.lane_id,
                "corridor_id": lane.corridor_id,
                "name": lane.name,
                "direction": lane.direction,
                "confidence": round(float(lane.confidence), 3),
                "avg_width_bev": round(float(lane.avg_width_bev), 2),
                "supporting_tracks": lane.supporting_tracks,
                "left_boundary_img": lane.left_boundary_img.tolist(),
                "right_boundary_img": lane.right_boundary_img.tolist(),
                "centerline_img": lane.centerline_img.tolist(),
            }
            for lane in temporal_model.active_lanes
        ],
    }

    challans_json = [
        row for row in violation_rows if row.get("challan_id")
    ]

    secondary_json = [asdict(d) for d in all_secondary]

    summary_json = {
        "video": args.video,
        "output_dir": args.output_dir,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "fps": fps,
        "processed_frames": len(all_assignments),
        "models": {
            "traffic": config.get("model", {}).get("path"),
            **{
                name: cfg.get("path")
                for name, cfg in config.get("secondary_models", {}).items()
                if isinstance(cfg, dict)
            },
        },
        "counts": {
            "tracks": len(tracks_json),
            "frames": len(all_assignments),
            "secondary_detections": len(secondary_json),
            "violations": len(violation_rows),
            "challans": len(challans_json),
            "lanes": len(temporal_model.active_lanes),
        },
        "json_outputs": {
            "summary": "summary.json",
            "frames": "frames.json",
            "tracks": "tracks.json",
            "lanes": "lanes.json",
            "secondary_detections": "secondary_detections.json",
            "signal_decisions": "signal_decisions.json",
            "violations": "violations.json",
            "challans": "challans.json",
            "integrated_report": "integrated_report.json",
        },
        "notes": [
            "Lane/outside-lane status is reported as lane_position_status and is not treated as an enforcement violation.",
            "Challans are generated only for configured violations such as wrong-way, speeding, red-light, helmet, weaving, and emergency obstruction.",
            "Entries marked review_required should be manually verified before any legal use.",
        ],
    }

    write_json(os.path.join(args.output_dir, "summary.json"), summary_json)
    write_json(os.path.join(args.output_dir, "frames.json"), all_assignments)
    write_json(os.path.join(args.output_dir, "tracks.json"), tracks_json)
    write_json(os.path.join(args.output_dir, "lanes.json"), lanes_json)
    write_json(os.path.join(args.output_dir, "secondary_detections.json"), secondary_json)
    write_json(os.path.join(args.output_dir, "signal_decisions.json"), signal_decisions)
    write_json(os.path.join(args.output_dir, "violations.json"), violation_rows)
    write_json(os.path.join(args.output_dir, "challans.json"), challans_json)

    write_json(
        os.path.join(args.output_dir, "integrated_report.json"),
        {
            **summary_json,
            "lane_model": lanes_json,
            "frames": all_assignments,
            "tracks": tracks_json,
            "secondary_detections": secondary_json,
            "signal_decisions": signal_decisions,
            "violations": violation_rows,
            "challans": challans_json,
        },
    )

    write_csv(
        os.path.join(args.output_dir, "violations.csv"),
        violation_rows,
        [
            "frame",
            "track_id",
            "class",
            "plate_number",
            "speed_kmh",
            "lane_id",
            "lane_name",
            "direction",
            "violations",
            "snapshot_path",
            "challan_id",
            "xai_justification",
            "slip_path",
            "slip_format",
            "evidence_path",
            "evidence_valid",
            "review_required",
            "plate_confidence",
            "xai_score",
            "xai_top_reason",
            "severity_level",
            "severity_score",
            "severity_reasons",
            "detection_confidence",
            "pdf_receipt",
            "html_receipt",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "secondary_detections.csv"),
        [asdict(d) for d in all_secondary],
        ["frame", "detector", "label", "confidence", "bbox"],
    )
    write_csv(
        os.path.join(args.output_dir, "signal_decisions.csv"),
        [
            {
                "frame": d["frame"],
                "reason": d["reason"],
                "active_phase": d["active_phase"]["phase_id"],
                "approach": d["active_phase"]["approach"],
                "green_seconds": d["active_phase"]["green_seconds"],
                "estimated_wait_seconds": d["active_phase"]["estimated_wait_seconds"],
                "queue_pressure": d["active_phase"]["queue_pressure"],
                "vehicle_count": d["active_phase"]["vehicle_count"],
                "emergency_priority": d["active_phase"]["emergency_priority"],
            }
            for d in signal_decisions
        ],
        [
            "frame",
            "reason",
            "active_phase",
            "approach",
            "green_seconds",
            "estimated_wait_seconds",
            "queue_pressure",
            "vehicle_count",
            "emergency_priority",
        ],
    )

    if last_annotated is not None:
        cv2.imwrite(os.path.join(args.output_dir, "preview.jpg"), last_annotated)

    elapsed = time.time() - start
    proc_fps = len(all_assignments) / elapsed if elapsed > 0 else 0.0
    print("=" * 72)
    print(f"Complete: {len(all_assignments)} frames in {elapsed:.1f}s ({proc_fps:.1f} fps)")
    print(f"Annotated video: {annotated_path}")
    if bev_writer is not None:
        print(f"Bird's-eye video: {bev_path}")
    print(f"Report: {os.path.join(args.output_dir, 'integrated_report.json')}")
    print(f"Violations CSV: {os.path.join(args.output_dir, 'violations.csv')}")
    print(f"Signal CSV: {os.path.join(args.output_dir, 'signal_decisions.csv')}")
    print(f"Violation slips: {slips_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
