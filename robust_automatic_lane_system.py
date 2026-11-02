#!/usr/bin/env python3
"""
robust_automatic_lane_system.py

Production-Grade Robust Automatic Lane Detection & Lane Assignment System
Engineered for Unstructured Indian Road Surveillance.

Key Capabilities:
1. INFERS VIRTUAL LANES WITHOUT PAINTED MARKINGS:
   Uses vehicle motion trajectories, road-tire contact points, and density valleys.
2. MULTI-CORRIDOR & PHYSICAL DIVIDER RESPECT:
   Separates distinct carriageways; lanes never cross physical medians/dividers.
3. INVERSE PERSPECTIVE MAPPING (IPM / BEV):
   Transforms road into bird's-eye view where lanes are parallel and width is metric-like.
4. TEMPORAL SMOOTHING & HYSTERESIS:
   Eliminates boundary fluttering via EMA control-point filtering.
5. PEDESTRIAN & PARKED VEHICLE EXCLUSION:
   Filters out pedestrians and stationary vehicles so they don't corrupt the lane model.
6. STABLE VEHICLE-TO-LANE ASSIGNMENT:
   Debounce buffer prevents rapid jumping between lanes; provides violation detection hooks.
7. COMPREHENSIVE VISUALIZATION & DATA EXPORT:
   Produces annotated video, standalone bird's-eye view video, and JSON metadata.

Usage:
  python robust_automatic_lane_system.py \
      --video "videos/Traffic congestion due to Metro work - South Delhi [-EnB6E3hApQ].mp4" \
      --max_frames 500 \
      --save_birdseye_video
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from modules import (
    TrackerPipeline,
    PerspectiveTransformer,
    SceneAnalyzer,
    TrajectoryClusterer,
    TemporalLaneModel,
    LaneAssigner,
    LaneVisualizer,
)


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Robust Automatic Lane Detection & Assignment for Indian Roads"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="videos/Traffic congestion due to Metro work - South Delhi [-EnB6E3hApQ].mp4",
        help="Path to input traffic video file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/lane_system_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save videos and JSON artifacts",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=500,
        help="Maximum frames to process (use -1 for full video)",
    )
    parser.add_argument(
        "--warmup_frames",
        type=int,
        default=40,
        help="Initial frames to accumulate trajectory evidence before fixing lane model",
    )
    parser.add_argument(
        "--update_interval",
        type=int,
        default=10,
        help="Frame interval at which to update lane clustering model",
    )
    parser.add_argument(
        "--save_birdseye_video",
        action="store_true",
        help="Save dedicated standalone bird's-eye view video",
    )
    args = parser.parse_args()

    # 1. Verify Inputs & Environment
    if not os.path.exists(args.video):
        sys.exit(f"Error: Video file not found: {args.video}")
    if not os.path.exists(args.config):
        sys.exit(f"Error: Config file not found: {args.config}")

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_yaml_config(args.config)

    # 2. Open Video Stream
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Error: Unable to open video stream at {args.video}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames_to_process = (
        total_video_frames if args.max_frames <= 0 else min(args.max_frames, total_video_frames)
    )

    print("=" * 70)
    print("ROBUST AUTOMATIC LANE DETECTION & LANE ASSIGNMENT SYSTEM")
    print(f"Video: {os.path.basename(args.video)}")
    print(f"Resolution: {frame_w}x{frame_h} @ {fps:.1f} fps")
    print(f"Frames to process: {n_frames_to_process} / {total_video_frames}")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    # 3. Initialize Modular Subsystems
    tracker = TrackerPipeline(config)
    transformer = PerspectiveTransformer(config, frame_w, frame_h)
    scene_analyzer = SceneAnalyzer(config, frame_w, frame_h)
    clusterer = TrajectoryClusterer(config)
    temporal_model = TemporalLaneModel(config)
    lane_assigner = LaneAssigner(config)
    visualizer = LaneVisualizer(config, frame_w, frame_h)

    # 4. Initialize Video Writers
    out_annotated_path = os.path.join(args.output_dir, "annotated_traffic_lanes.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_annotated = cv2.VideoWriter(
        out_annotated_path, fourcc, fps, (frame_w, frame_h)
    )

    writer_bev = None
    if args.save_birdseye_video:
        out_bev_path = os.path.join(args.output_dir, "birdseye_debug.mp4")
        writer_bev = cv2.VideoWriter(
            out_bev_path, fourcc, fps, (transformer.ipm_w, transformer.ipm_h)
        )

    # Accumulator for all frame assignments
    all_frame_assignments: List[Dict[str, Any]] = []
    total_violations_counter = 0

    # 5. Main Processing Loop
    start_time = time.time()
    pbar = tqdm(total=n_frames_to_process, desc="Processing Video", unit="frame")

    frame_idx = 0
    while frame_idx < n_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        # Step A: Multi-Object Detection & Tracking
        current_vehicles, track_summaries, current_pedestrians = tracker.process_frame(
            frame, frame_idx
        )

        # Step B: Accumulate Scene Occupancy
        scene_analyzer.accumulate_occupancy(current_vehicles, current_pedestrians)

        # Step C: Dynamic Perspective & Scene Geometry Updates
        clean_moving_tracks = tracker.get_clean_moving_tracks()

        if frame_idx >= args.warmup_frames and (frame_idx % args.update_interval == 0 or not temporal_model.is_initialized):
            # 1. Update Vanishing Point from moving tracks
            track_lines = []
            for t in clean_moving_tracks:
                pts = t.history_bottom_centers
                if len(pts) >= 2:
                    track_lines.append((pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]))

            vp = transformer.estimate_vanishing_point(track_lines)

            # 2. Discover Corridors & Dividers
            corridors, dividers = scene_analyzer.analyze_corridors_and_dividers(
                clean_moving_tracks, vp, transformer=transformer
            )

            # 3. Infer Virtual Lanes for All Corridors in the Scene
            all_new_lanes = []
            for corr in corridors:
                corr_trans = PerspectiveTransformer(config, frame_w, frame_h)
                corr_trans.update_homography_from_corridor(
                    corr.x_left_bottom,
                    corr.x_right_bottom,
                    corr.x_left_top,
                    corr.x_right_top,
                )
                c_lanes = clusterer.infer_lanes_for_corridor(
                    corr, clean_moving_tracks, corr_trans
                )
                all_new_lanes.extend(c_lanes)

            # Re-index lane IDs sequentially
            for l_idx, l in enumerate(all_new_lanes):
                l.lane_id = l_idx

            # 4. Smooth Combined Lane Model Temporally
            temporal_model.update(all_new_lanes, frame_idx)

        # Step D: Vehicle-to-Lane Assignment
        active_lanes = temporal_model.active_lanes
        assignments = lane_assigner.assign_vehicles(
            current_vehicles, track_summaries, active_lanes
        )

        # Record assignments
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
                    "confidence": a.lane_conf,
                    "dist_from_center_px": a.dist_from_center_px,
                    "direction": a.direction,
                    "contact_point": [round(c, 1) for c in a.contact_point],
                    "is_violating": a.is_violating,
                    "violations": a.violation_types,
                }
                for a in assignments
            ],
        }
        all_frame_assignments.append(frame_record)
        total_violations_counter += sum(1 for a in assignments if a.is_violating)

        # Step E: Visualization & Video Writing
        annotated_frame = visualizer.render_frame(
            frame=frame,
            active_lanes=active_lanes,
            dividers=scene_analyzer.dividers,
            assignments=assignments,
            current_vehicles=current_vehicles,
            track_summaries=track_summaries,
            transformer=transformer,
            frame_idx=frame_idx,
            fps=fps,
        )
        writer_annotated.write(annotated_frame)

        if writer_bev is not None:
            bev_canvas = visualizer.render_birdseye_canvas(
                active_lanes, scene_analyzer.dividers, track_summaries, transformer
            )
            writer_bev.write(bev_canvas)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer_annotated.release()
    if writer_bev is not None:
        writer_bev.release()

    elapsed = time.time() - start_time
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 70)
    print(f"Processing Complete! Processed {frame_idx} frames in {elapsed:.1f}s ({proc_fps:.1f} fps)")
    print(f"Saved annotated video: {out_annotated_path}")
    if args.save_birdseye_video:
        print(f"Saved bird's-eye debug video: {out_bev_path}")

    # 6. Export Metadata & Compatibility Artifacts
    # A. lane_boundaries.json (100% compatible with lane_utils.py & verify_lanes.py)
    boundaries_json = temporal_model.export_boundaries_json(frame_w, frame_h)
    out_boundaries_json_path = os.path.join(args.output_dir, "lane_boundaries.json")
    with open(out_boundaries_json_path, "w", encoding="utf-8") as f:
        json.dump(boundaries_json, f, indent=2, cls=NumpyJSONEncoder)
    print(f"Saved compatible lane boundaries: {out_boundaries_json_path}")

    # B. lane_model.json (comprehensive details)
    lane_model_data = {
        "video": os.path.basename(args.video),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "num_lanes": len(temporal_model.active_lanes),
        "global_confidence": round(temporal_model.global_confidence, 3),
        "vanishing_point": [round(c, 1) for c in transformer.vanishing_point],
        "lanes": [
            {
                "lane_id": l.lane_id,
                "name": l.name,
                "direction": l.direction,
                "confidence": round(l.confidence, 3),
                "avg_width_bev": round(l.avg_width_bev, 1),
                "supporting_tracks": l.supporting_tracks,
                "left_boundary_img": [[round(c, 1) for c in pt] for pt in l.left_boundary_img],
                "right_boundary_img": [[round(c, 1) for c in pt] for pt in l.right_boundary_img],
                "centerline_img": [[round(c, 1) for c in pt] for pt in l.centerline_img],
            }
            for l in temporal_model.active_lanes
        ],
        "dividers": [
            {
                "divider_id": d.divider_id,
                "x_left_bot": round(d.x_left_bot, 1),
                "x_right_bot": round(d.x_right_bot, 1),
                "x_left_top": round(d.x_left_top, 1),
                "x_right_top": round(d.x_right_top, 1),
            }
            for d in scene_analyzer.dividers
        ],
    }
    out_lane_model_path = os.path.join(args.output_dir, "lane_model.json")
    with open(out_lane_model_path, "w", encoding="utf-8") as f:
        json.dump(lane_model_data, f, indent=2, cls=NumpyJSONEncoder)
    print(f"Saved lane model details: {out_lane_model_path}")

    # C. vehicle_tracks.json
    clean_tracks = tracker.get_clean_moving_tracks()
    tracks_data = [
        {
            "track_id": t.track_id,
            "class": t.cls_name,
            "direction": t.direction,
            "displacement_px": round(t.total_displacement, 1),
            "num_observations": len(t.observations),
            "start_point": [round(c, 1) for c in t.observations[0].bottom_center],
            "end_point": [round(c, 1) for c in t.observations[-1].bottom_center],
        }
        for t in clean_tracks
    ]
    out_tracks_path = os.path.join(args.output_dir, "vehicle_tracks.json")
    with open(out_tracks_path, "w", encoding="utf-8") as f:
        json.dump(tracks_data, f, indent=2, cls=NumpyJSONEncoder)
    print(f"Saved vehicle tracks: {out_tracks_path} ({len(tracks_data)} clean moving tracks)")

    # D. lane_assignments.json
    out_assignments_path = os.path.join(args.output_dir, "lane_assignments.json")
    with open(out_assignments_path, "w", encoding="utf-8") as f:
        json.dump(all_frame_assignments, f, indent=2, cls=NumpyJSONEncoder)
    print(f"Saved lane assignments log: {out_assignments_path}")

    # E. Save high-resolution verification sample image
    if temporal_model.active_lanes and frame_idx > 0:
        sample_verify_path = os.path.join(args.output_dir, "lane_verify_preview.jpg")
        cv2.imwrite(sample_verify_path, annotated_frame)
        print(f"Saved verification preview frame: {sample_verify_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
