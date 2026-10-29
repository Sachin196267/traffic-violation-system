#!/usr/bin/env python3
"""Run synthetic smart-signal scenarios without processing a video."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.lane_assigner import VehicleAssignment
from modules.smart_signal_controller import SmartSignalController


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_assignment(track_id, cls_name, lane_id, direction, x, y, violating=False):
    return VehicleAssignment(
        track_id=track_id,
        cls_name=cls_name,
        lane_id=lane_id,
        lane_name=f"Lane {lane_id + 1}" if lane_id is not None else "Outside",
        lane_conf=0.8,
        dist_from_center_px=0.0,
        direction=direction,
        contact_point=(x, y),
        is_violating=violating,
        violation_types=["QUEUE_BLOCKING"] if violating else [],
    )


def build_scenario(name: str, frame_w: int, frame_h: int):
    scenarios = {
        "balanced": [
            ("car", 0, "inbound", 900, 1700, False),
            ("motorcycle", 0, "inbound", 1050, 1800, False),
            ("bus", 1, "outbound", 2300, 900, False),
            ("car", 1, "outbound", 2450, 1100, False),
        ],
        "north_south_heavy": [
            ("bus", 0, "inbound", 900, 1700, False),
            ("truck", 0, "inbound", 1020, 1780, False),
            ("car", 0, "inbound", 1130, 1840, False),
            ("car", 0, "inbound", 1250, 1900, False),
            ("motorcycle", 0, "inbound", 1340, 1940, False),
            ("car", 1, "outbound", 2600, 900, False),
        ],
        "east_west_heavy": [
            ("car", 1, "outbound", 2300, 900, False),
            ("car", 1, "outbound", 2450, 980, False),
            ("bus", 1, "outbound", 2600, 1060, False),
            ("truck", 1, "outbound", 2750, 1140, False),
            ("motorcycle", 1, "outbound", 2880, 1220, False),
            ("car", 0, "inbound", 900, 1800, False),
        ],
        "violation_pressure": [
            ("car", None, "unknown", 700, 1900, True),
            ("bus", None, "unknown", 900, 1850, True),
            ("truck", None, "unknown", 1150, 1800, True),
            ("car", 1, "outbound", 2500, 950, False),
        ],
    }
    rows = scenarios.get(name)
    if rows is None:
        raise SystemExit(f"Unknown scenario: {name}")
    return [
        make_assignment(i + 1, cls_name, lane_id, direction, x, y, violating)
        for i, (cls_name, lane_id, direction, x, y, violating) in enumerate(rows)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/integrated_system_config.yaml")
    parser.add_argument(
        "--scenario",
        default="north_south_heavy",
        choices=["balanced", "north_south_heavy", "east_west_heavy", "violation_pressure"],
    )
    parser.add_argument("--ambulance", action="store_true")
    args = parser.parse_args()

    frame_w, frame_h = 3840, 2160
    config = load_config(args.config)
    controller = SmartSignalController(config, frame_w, frame_h)
    assignments = build_scenario(args.scenario, frame_w, frame_h)
    secondary = []
    if args.ambulance:
        secondary.append(
            SimpleNamespace(
                detector="emergency_vehicle",
                label="ambulance",
                confidence=0.91,
                bbox=[400.0, 1300.0, 900.0, 1900.0],
            )
        )

    decision = controller.update(
        frame_idx=0,
        assignments=assignments,
        secondary_detections=secondary,
        lane_confidence=0.82,
    )
    print(json.dumps({
        "scenario": args.scenario,
        "ambulance": args.ambulance,
        "decision": {
            "reason": decision.reason,
            "active_phase": asdict(decision.active_phase),
            "approach_state": decision.approach_state,
            "best_scenarios": [
                asdict(s) for s in sorted(decision.scenarios, key=lambda s: s.estimated_wait_seconds)[:5]
            ],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
