#!/usr/bin/env python3
"""
Master End-to-End Traffic Violation, Digital Twin Visualizer & e-Challan Pipeline.
Tailored for the Bombay intersection surveillance stream (https://youtu.be/zxxfvP8-lrU).
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import yaml
import cv2
import numpy as np
from datetime import datetime
from tqdm import tqdm

from detection_and_tracking import TrackerPipeline
from violation_detector import ViolationDetector
from anpr_engine import ANPREngine
from xai_and_gradcam import XAIExplainer
from digital_twin_engine import DigitalTwinEngine
from echallan_pdf_generator import EChallanPDFGenerator

def run_pipeline(
    config_path: str = "traffic_violation/final_video_testing/config.yaml",
    max_frames: int = 150,
    start_frame: int = 0
):
    start_time = time.time()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    video_path = cfg["video"]["input_path"]
    out_video_path = cfg["video"]["output_path"]
    evidence_dir = "traffic_violation/final_video_testing/evidence"
    challans_dir = "traffic_violation/final_video_testing/challans"
    os.makedirs(evidence_dir, exist_ok=True)
    os.makedirs(challans_dir, exist_ok=True)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video {video_path} not found. Please run download_video.py first.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video stream at {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_to_process = total_video_frames - start_frame if max_frames < 0 else min(max_frames, total_video_frames - start_frame)

    print("=" * 75)
    print("TRAFFIC VIOLATION ENFORCEMENT & 3D DIGITAL TWIN PIPELINE")
    print(f"Source: {video_path} ({w}x{h} @ {fps:.1f} fps)")
    print(f"Frames to analyze: {frames_to_process} (start frame: {start_frame})")
    print(f"Output Video: {out_video_path}")
    print(f"Challans Directory: {challans_dir}")
    print("=" * 75)

    # Initialize Modules
    device = cfg["models"].get("device", "cpu")
    tracker = TrackerPipeline(model_path=cfg["models"]["primary_detector"], device=device)
    detector = ViolationDetector(config=cfg)
    anpr = ANPREngine(model_path=cfg["models"]["license_plate_detector"], device=device)
    xai = XAIExplainer(config=cfg)
    digital_twin = DigitalTwinEngine(config=cfg)
    pdf_gen = EChallanPDFGenerator(output_dir=challans_dir)

    # Video Writer for Side-by-Side Composite
    twin_w = digital_twin.canvas_w
    composite_w = w + twin_w
    composite_h = h
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (composite_w, composite_h))

    challan_counter = 0
    all_challans = []
    # Persistent active violation visual states: {track_id: (violation_list, expiry_frame)}
    active_visual_alerts: dict[int, tuple[list[str], int]] = {}

    pbar = tqdm(total=frames_to_process, desc="Enforcing & Mirroring")

    current_f = start_frame
    while current_f < start_frame + frames_to_process:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # 1. ByteTrack Object Tracking
        tracked = tracker.track_frame(frame, current_f, conf_thresh=0.25)

        # 2. Multi-Violation Evaluation
        new_events, signal_state, countdown = detector.analyze_frame(frame, current_f, fps, tracked)

        # 3. Handle Active Visual Alerts (persist for 30 frames for smooth visualization)
        for ev in new_events:
            tid = ev["track_id"]
            active_visual_alerts[tid] = (ev["violations"], current_f + 35)

        # Clean expired alerts
        for tid in list(active_visual_alerts.keys()):
            if current_f > active_visual_alerts[tid][1]:
                del active_visual_alerts[tid]

        # Active violations dictionary for digital twin: {tid: [violation_types]}
        active_for_twin = {tid: val[0] for tid, val in active_visual_alerts.items()}

        # 4. Process New Violation Events -> Generate e-Challan & Evidence Dossier
        for ev in new_events:
            challan_counter += 1
            tid = ev["track_id"]
            v_types = ev["violations"]
            v_crop = ev["vehicle_crop"]
            dt_now = datetime.now().strftime("%d-%b-%Y %H:%M:%S IST")
            challan_id = f"MH-TVS-{datetime.now().strftime('%Y%m%d')}-{current_f:05d}-{tid:04d}"

            # ANPR Extraction
            plate_info = anpr.extract_plate(v_crop, tid)
            plate_num = plate_info["plate_number"]
            plate_crop = plate_info["plate_crop"]

            # Save Evidence Images
            scene_path = os.path.join(evidence_dir, f"{challan_id}_scene.jpg")
            vehicle_path = os.path.join(evidence_dir, f"{challan_id}_vehicle.jpg")
            plate_path = os.path.join(evidence_dir, f"{challan_id}_plate.jpg")
            gradcam_path = os.path.join(evidence_dir, f"{challan_id}_gradcam.jpg")

            # Save scene with highlight
            annotated_scene = frame.copy()
            bx, by, bw, bh = ev["bbox"]
            cv2.rectangle(annotated_scene, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
            cv2.putText(annotated_scene, f"VIOLATION: {v_types[0]}", (bx, max(20, by - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.imwrite(scene_path, annotated_scene)

            # Save vehicle crop
            if v_crop is not None and v_crop.size > 0:
                cv2.imwrite(vehicle_path, v_crop)

            # Save plate crop
            if plate_crop is not None and plate_crop.size > 0:
                cv2.imwrite(plate_path, plate_crop)
            else:
                cv2.imwrite(plate_path, v_crop)

            # Generate Grad-CAM Heatmap
            gradcam_img = xai.generate_gradcam_heatmap(v_crop, v_types[0])
            cv2.imwrite(gradcam_path, gradcam_img)

            # XAI Reasoning & Severity Index
            confidences = {v: 0.92 for v in v_types}
            ev_metrics = dict(ev.get("evidence_metrics", {}))
            ev_metrics["bbox"] = ev.get("bbox")
            ev_metrics["contact_point"] = ev.get("contact_point")
            ev_metrics["speed_kmh"] = ev.get("speed_kmh")
            ev_metrics["is_xywh"] = True

            xai_data = xai.generate_reasoning_matrix(
                track_id=tid,
                vehicle_class=ev["class"],
                violations=v_types,
                confidences=confidences,
                evidence_metrics=ev_metrics
            )
            sev_data = xai.compute_severity(
                violations=v_types,
                speed_kmh=ev["speed_kmh"],
                red_elapsed_sec=ev_metrics.get("red_elapsed_sec", 0.0),
                passenger_count=ev_metrics.get("passenger_count", 1)
            )

            # Generate PDF
            pdf_path = pdf_gen.generate_challan(
                challan_id=challan_id,
                timestamp_str=dt_now,
                plate_number=plate_num,
                vehicle_class=ev["class"],
                speed_kmh=ev["speed_kmh"],
                violations=v_types,
                severity_data=sev_data,
                xai_data=xai_data,
                scene_img_path=scene_path,
                vehicle_crop_path=vehicle_path,
                plate_crop_path=plate_path,
                gradcam_path=gradcam_path
            )

            all_challans.append({
                "challan_id": challan_id,
                "frame": current_f,
                "track_id": tid,
                "class": ev["class"],
                "plate": plate_num,
                "violations": v_types,
                "severity": sev_data["level"],
                "score": sev_data["score"],
                "pdf": pdf_path,
                "primary_justification": xai_data.get("primary_justification", "")
            })

        # 5. Render Real-World Camera Frame Overlay
        cam_canvas = frame.copy()

        # Draw Imaginary Virtual Lane Corridor
        poly_pts = np.array(cfg["spatial"]["virtual_lane_polygon"], dtype=np.int32)
        poly_overlay = cam_canvas.copy()
        cv2.fillPoly(poly_overlay, [poly_pts], (255, 220, 0))
        cam_canvas = cv2.addWeighted(poly_overlay, 0.18, cam_canvas, 0.82, 0)
        cv2.polylines(cam_canvas, [poly_pts], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # Draw Stop Line
        stop_p1 = tuple(cfg["spatial"]["stop_line"]["p1"])
        stop_p2 = tuple(cfg["spatial"]["stop_line"]["p2"])
        stop_col = (0, 0, 255) if signal_state == "RED" else (255, 255, 255)
        cv2.line(cam_canvas, stop_p1, stop_p2, stop_col, 3, lineType=cv2.LINE_AA)

        # Draw Vehicle Bounding Boxes & Warning Badges
        for v in tracked:
            tid = v["track_id"]
            bx, by, bw, bh = v["bbox"]
            is_violating = tid in active_visual_alerts
            box_col = (0, 0, 255) if is_violating else (0, 255, 120)

            cv2.rectangle(cam_canvas, (bx, by), (bx + bw, by + bh), box_col, 2 if not is_violating else 3)
            # Label
            lbl = f"#{tid} {v['class']} {v['speed_kmh']:.0f}km/h"
            cv2.rectangle(cam_canvas, (bx, max(0, by - 18)), (bx + len(lbl) * 8, max(18, by)), (20, 24, 30), -1)
            cv2.putText(cam_canvas, lbl, (bx + 2, max(14, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            if is_violating:
                v_name = active_visual_alerts[tid][0][0]
                v_badge = f"ALERT: {v_name.replace('_', ' ')}"
                cv2.rectangle(cam_canvas, (bx, by + bh), (bx + len(v_badge) * 9, by + bh + 20), (0, 0, 200), -1)
                cv2.putText(cam_canvas, v_badge, (bx + 4, by + bh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Camera HUD Header
        cv2.rectangle(cam_canvas, (0, 0), (w, 40), (14, 18, 25), -1)
        hud_title = f"MUMBAI ITMS LIVE SURVEILLANCE | FRAME #{current_f:05d}"
        cv2.putText(cam_canvas, hud_title, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

        # 6. Render Digital Twin Canvas
        twin_base = digital_twin.render_base_canvas(signal_state, countdown)
        twin_frame = digital_twin.update_and_render_vehicles(twin_base, current_f, tracked, active_for_twin)

        # 7. Stack Side-by-Side Composite
        composite = digital_twin.composite_side_by_side(cam_canvas, twin_frame)
        writer.write(composite)

        current_f += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    proc_fps = frames_to_process / elapsed if elapsed > 0 else 0.0

    print("=" * 75)
    print("PIPELINE PROCESSING COMPLETE!")
    print(f"Total Frames Processed: {frames_to_process} in {elapsed:.1f}s ({proc_fps:.1f} fps)")
    print(f"Rendered Output Video: {out_video_path}")
    print(f"Total e-Challan Notices Generated: {len(all_challans)}")
    print(f"e-Challan PDFs Location: {challans_dir}/")
    for c in all_challans:
        print(f"  -> [{c['severity']} ({c['score']}/100)] {c['challan_id']} | Reg: {c['plate']} | {','.join(c['violations'])}")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="traffic_violation/final_video_testing/config.yaml")
    parser.add_argument("--max_frames", type=int, default=150, help="-1 for entire video")
    parser.add_argument("--start_frame", type=int, default=0)
    args = parser.parse_args()
    run_pipeline(args.config, args.max_frames, args.start_frame)
