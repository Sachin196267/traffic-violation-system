import os
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.violation_slip_generator import ViolationSlipGenerator, create_evidence_composite, SlipResult
from modules.lane_assigner import VehicleAssignment, LaneAssigner
from modules.xai_explainer import ViolationXAIExplainer
from modules.tracker_pipeline import TrackObservation, TrackSummary
from modules.temporal_lane_model import InferredLane
from modules.scene_analyzer import RoadCorridor

def test_evidence_composite():
    print("[1/4] Testing Evidence Composite Generation...")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.rectangle(frame, (300, 200), (1600, 1080), (70, 70, 70), -1)
    bbox = (800.0, 500.0, 950.0, 750.0)
    cv2.rectangle(frame, (800, 500), (950, 750), (30, 30, 200), -1)
    trajectory = [(850, 400), (860, 450), (875, 520), (875, 750)]
    contact_pt = (875.0, 750.0)

    composite = create_evidence_composite(
        frame=frame,
        bbox=bbox,
        track_id=42,
        cls_name="car",
        violation_types=["WRONG_WAY_DRIVING"],
        contact_point=contact_pt,
        trajectory=trajectory,
        target_width=1200,
        target_height=560,
    )
    assert composite is not None, "Composite image is None"
    assert composite.shape == (560, 1200, 3), f"Unexpected shape {composite.shape}"
    print("  -> Evidence composite generated successfully (1200x560x3).")
    return frame, bbox, contact_pt, trajectory

def test_pdf_generation(frame, bbox, contact_pt, trajectory):
    print("[2/4] Testing PDF Slip Generation & Sizing...")
    out_dir = "outputs/test_challan_output"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    generator = ViolationSlipGenerator(out_dir)

    assignment = VehicleAssignment(
        track_id=42,
        cls_name="car",
        lane_id=0,
        lane_name="Lane 1",
        lane_conf=0.92,
        dist_from_center_px=-12.5,
        direction="inbound",
        contact_point=contact_pt,
        is_violating=True,
        violation_types=["WRONG_WAY_DRIVING"],
    )

    xai = ViolationXAIExplainer().explain(
        assignment=assignment,
        lane_confidence=0.92,
        signal_reason="Normal signal cycle",
        emergency_priority=False,
    )

    signal_record = {
        "active_phase": {
            "phase_id": 1,
            "label": "North-South Green",
            "green_seconds": 25,
            "emergency_priority": False,
        }
    }

    slip = generator.generate(
        assignment=assignment,
        frame_idx=120,
        snapshot_path="",
        xai=xai,
        signal_record=signal_record,
        secondary_detections=[],
        video_name="test_traffic.mp4",
        frame=frame,
        bbox=bbox,
        trajectory=trajectory,
    )

    assert slip.format == "pdf", f"Expected PDF format, got {slip.format}"
    assert os.path.exists(slip.path), f"PDF file not found at {slip.path}"
    pdf_size = os.path.getsize(slip.path)
    print(f"  -> Generated PDF size: {pdf_size / 1024:.1f} KB (Path: {slip.path})")
    assert 20 * 1024 <= pdf_size <= 400 * 1024, f"PDF size {pdf_size} bytes is out of expected range (20KB - 400KB)"

    with open(slip.path, "rb") as f:
        pdf_bytes = f.read()
    page_count = sum([pdf_bytes.count(b"/Type /Page\n"), pdf_bytes.count(b"/Type /Page\r"), pdf_bytes.count(b"/Type /Page ")])
    print(f"  -> Total pages in PDF: {page_count}")
    assert page_count == 1, f"PDF should be strictly 1 page, found {page_count} pages!"
    print("  -> PDF single-page validation passed.")

def test_lane_assigner_safeguards():
    print("[3/4] Testing Lane Assigner Safeguards...")
    config = {
        "lane_assignment": {
            "lateral_tolerance_px": 25.0,
            "outside_min_lane_confidence": 0.65,
            "outside_min_track_observations": 8,
            "outside_min_displacement_px": 25.0,
            "outside_required_votes": 3,
            "outside_margin_multiplier": 2.5,
        },
        "temporal_model": {"hysteresis_frames": 4},
    }
    assigner = LaneAssigner(config)

    lane_poly = np.array([[600, 1000], [900, 1000], [850, 400], [700, 400]], dtype=np.int32)
    centerline = np.array([[750, 1000], [775, 400]], dtype=np.int32)
    dummy_bev = np.zeros((10, 2), dtype=np.float32)
    dummy_img = np.zeros((10, 2), dtype=np.float32)
    active_lane = InferredLane(
        lane_id=0,
        corridor_id=0,
        name="Lane 1",
        direction="inbound",
        left_boundary_bev=dummy_bev,
        right_boundary_bev=dummy_bev,
        centerline_bev=dummy_bev,
        left_boundary_img=dummy_img,
        right_boundary_img=dummy_img,
        centerline_img=centerline,
        polygon_img=lane_poly,
        avg_width_bev=3.5,
        confidence=0.85,
        supporting_tracks=10,
    )

    # Drivable road corridor: wider area enclosing the lane and surrounding asphalt
    corridor_poly = np.array([[400, 1050], [1200, 1050], [1000, 350], [600, 350]], dtype=np.int32)
    corridor = RoadCorridor(
        corridor_id=0,
        name="Main Carriageway",
        dominant_direction="inbound",
        x_left_bottom=400,
        x_right_bottom=1200,
        x_left_top=600,
        x_right_top=1000,
        polygon=corridor_poly,
    )

    # Test Case 1: Vehicle on asphalt inside corridor but outside the single virtual lane (x=1050, y=800)
    obs1 = TrackObservation(
        track_id=101,
        cls_name="car",
        bbox=(1000, 700, 1100, 800),
        center=(1050, 750),
        bottom_center=(1050, 800),
        width=100,
        height=100,
        confidence=0.9,
        frame_idx=20,
    )
    summary1 = TrackSummary(track_id=101, cls_name="car")
    for f_idx in range(15):
        summary1.observations.append(TrackObservation(
            track_id=101, cls_name="car", bbox=(1000, 700, 1100, 800), center=(1050, 750),
            bottom_center=(1050, 800 + f_idx * 5), width=100, height=100, confidence=0.9, frame_idx=f_idx
        ))
    summary1.total_displacement = 75.0
    summary1.direction = "inbound"

    assignments1 = assigner.assign_vehicles(
        vehicles=[obs1],
        track_summaries={101: summary1},
        active_lanes=[active_lane],
        corridors=[corridor],
        frame_shape=(1080, 1920),
    )
    assert not assignments1[0].is_violating, f"False positive! Car in valid corridor flagged as violating: {assignments1[0].violation_types}"
    print("  -> Case 1 PASSED: Vehicle inside corridor polygon protected from false outside-road violation.")

    # Test Case 2: Vehicle near top horizon edge (y=180 in 1080p, y/fh = 0.16 < 0.22)
    obs2 = TrackObservation(
        track_id=102,
        cls_name="car",
        bbox=(200, 130, 280, 180),
        center=(240, 155),
        bottom_center=(240, 180),
        width=80,
        height=50,
        confidence=0.85,
        frame_idx=20,
    )
    summary2 = TrackSummary(track_id=102, cls_name="car")
    summary2.observations = [obs2] * 12
    summary2.total_displacement = 60.0
    summary2.direction = "inbound"

    assignments2 = assigner.assign_vehicles(
        vehicles=[obs2],
        track_summaries={102: summary2},
        active_lanes=[active_lane],
        corridors=[corridor],
        frame_shape=(1080, 1920),
    )
    assert not assignments2[0].is_violating, "False positive! Vehicle near horizon edge was ticketed!"
    print("  -> Case 2 PASSED: Vehicle near image boundary protected.")

    # Test Case 3: Genuine off-road vehicle (far outside corridor and lane)
    # should be reported as lane position status only, not as an enforcement violation.
    obs3 = TrackObservation(
        track_id=103,
        cls_name="car",
        bbox=(100, 600, 200, 700),
        center=(150, 650),
        bottom_center=(150, 700),
        width=100,
        height=100,
        confidence=0.95,
        frame_idx=20,
    )
    summary3 = TrackSummary(track_id=103, cls_name="car")
    for f in range(15):
        summary3.observations.append(TrackObservation(
            track_id=103, cls_name="car", bbox=(100, 600, 200, 700), center=(150, 650),
            bottom_center=(150, 700 + f * 4), width=100, height=100, confidence=0.9, frame_idx=f
        ))
    summary3.total_displacement = 60.0
    summary3.direction = "inbound"

    # Send enough votes to trigger the outside-corridor debounce
    for _ in range(assigner.outside_required_votes):
        assignments3 = assigner.assign_vehicles(
            vehicles=[obs3],
            track_summaries={103: summary3},
            active_lanes=[active_lane],
            corridors=[corridor],
            frame_shape=(1080, 1920),
        )
    assert not assignments3[0].is_violating, f"Outside-lane status should not issue a violation: {assignments3[0].violation_types}"
    assert assignments3[0].lane_position_status == "outside_drivable_corridor", assignments3[0].lane_position_status
    print("  -> Case 3 PASSED: Outside-corridor status reported without issuing a violation.")

def test_cooldown_debouncing():
    print("[4/4] Testing Per-Track Challan Cooldown Debouncing...")
    issued_challans = {}
    cooldown_frames = 300
    generated_slips = []

    # Simulate 50 consecutive frames where track 55 violates WRONG_WAY_DRIVING
    for frame_idx in range(100, 150):
        assignment = VehicleAssignment(
            track_id=55,
            cls_name="truck",
            lane_id=None,
            lane_name="Outside Lane",
            lane_conf=0.7,
            dist_from_center_px=-90.0,
            direction="inbound",
            contact_point=(120.0, 650.0),
            is_violating=True,
            violation_types=["WRONG_WAY_DRIVING"],
        )

        violation_sig = "|".join(sorted(assignment.violation_types))
        last_issued = issued_challans.get((assignment.track_id, violation_sig), -999999)
        if (frame_idx - last_issued) < cooldown_frames:
            continue
        issued_challans[(assignment.track_id, violation_sig)] = frame_idx
        generated_slips.append((frame_idx, assignment.track_id))

    assert len(generated_slips) == 1, f"Expected exactly 1 slip, but got {len(generated_slips)} slips!"
    print(f"  -> Successfully debounced 50 consecutive frames into exactly {len(generated_slips)} challan slip at frame {generated_slips[0][0]}.")

if __name__ == "__main__":
    frame, bbox, contact_pt, trajectory = test_evidence_composite()
    test_pdf_generation(frame, bbox, contact_pt, trajectory)
    test_lane_assigner_safeguards()
    test_cooldown_debouncing()
    print("\nALL 4 PIPELINE TESTS PASSED!")
