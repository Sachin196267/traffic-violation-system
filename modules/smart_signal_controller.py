"""
Adaptive traffic signal controller.

The controller estimates approach demand from tracked vehicles, tests multiple
signal timing scenarios, and selects the plan with the lowest projected waiting
time. Emergency vehicle detections override the optimizer and force green for
the corresponding approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .lane_assigner import VehicleAssignment


VEHICLE_WEIGHTS = {
    "motorcycle": 0.45,
    "bicycle": 0.35,
    "autorickshaw": 0.75,
    "car": 1.00,
    "vehicle fallback": 1.00,
    "truck": 2.20,
    "bus": 2.50,
}


@dataclass
class SignalPhase:
    phase_id: str
    label: str
    approach: str
    green_seconds: int
    estimated_wait_seconds: float
    queue_pressure: float
    vehicle_count: int
    emergency_priority: bool = False


@dataclass
class SignalDecision:
    frame: int
    active_phase: SignalPhase
    scenarios: List[SignalPhase] = field(default_factory=list)
    approach_state: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reason: str = "adaptive_min_wait"


class SmartSignalController:
    def __init__(self, config: Dict[str, Any], frame_w: int, frame_h: int):
        self.config = config
        self.frame_w = frame_w
        self.frame_h = frame_h
        signal_cfg = config.get("smart_signal", {})

        self.enabled = signal_cfg.get("enabled", True)
        self.min_green = int(signal_cfg.get("min_green_seconds", 10))
        self.max_green = int(signal_cfg.get("max_green_seconds", 60))
        self.yellow = int(signal_cfg.get("yellow_seconds", 3))
        self.all_red = int(signal_cfg.get("all_red_seconds", 1))
        self.saturation_rate = float(signal_cfg.get("saturation_rate_vehicles_per_second", 0.55))
        self.update_interval = int(signal_cfg.get("update_interval_frames", 15))
        self.history_alpha = float(signal_cfg.get("queue_ema_alpha", 0.35))
        self.emergency_hold = int(signal_cfg.get("emergency_hold_frames", 90))
        self.default_cycle = int(signal_cfg.get("default_cycle_seconds", 90))
        self.approaches = signal_cfg.get("approaches", ["north_south", "east_west"])

        self.queue_history = {
            approach: {"pressure": 0.0, "vehicles": 0.0, "violations": 0.0}
            for approach in self.approaches
        }
        self.current_decision: Optional[SignalDecision] = None
        self.last_update_frame = -10**9
        self.emergency_until_frame = -1
        self.emergency_approach: Optional[str] = None

    def update(
        self,
        frame_idx: int,
        assignments: List[VehicleAssignment],
        secondary_detections: List[Any],
        lane_confidence: float,
    ) -> SignalDecision:
        if not self.enabled:
            return self._fixed_time_decision(frame_idx)

        approach_state = self._estimate_approaches(assignments, lane_confidence)
        emergency_approach = self._detect_emergency_approach(secondary_detections)
        if emergency_approach:
            self.emergency_approach = emergency_approach
            self.emergency_until_frame = max(self.emergency_until_frame, frame_idx + self.emergency_hold)

        if (
            self.current_decision is not None
            and frame_idx - self.last_update_frame < self.update_interval
            and frame_idx > self.emergency_until_frame
        ):
            self.current_decision.approach_state = approach_state
            return self.current_decision

        if frame_idx <= self.emergency_until_frame and self.emergency_approach:
            decision = self._emergency_decision(frame_idx, approach_state, self.emergency_approach)
        else:
            decision = self._optimize(frame_idx, approach_state)

        self.current_decision = decision
        self.last_update_frame = frame_idx
        return decision

    def render_overlay(self, frame: np.ndarray, decision: Optional[SignalDecision]) -> np.ndarray:
        if decision is None:
            return frame

        panel_w = 360
        panel_h = 132
        x0 = 18
        y0 = 70
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (10, 15, 18), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (70, 90, 100), 1)

        phase = decision.active_phase
        title = "SMART SIGNAL CONTROLLER"
        cv2.putText(frame, title, (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        color = (0, 255, 0) if decision.reason != "emergency_priority" else (0, 80, 255)
        cv2.circle(frame, (x0 + 24, y0 + 56), 12, color, -1)
        cv2.putText(
            frame,
            f"GREEN: {phase.label}  {phase.green_seconds}s",
            (x0 + 46, y0 + 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 255, 230),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Projected wait: {phase.estimated_wait_seconds:.1f}s | pressure: {phase.queue_pressure:.1f}",
            (x0 + 12, y0 + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 230, 255),
            1,
            cv2.LINE_AA,
        )
        reason = "Emergency priority" if decision.reason == "emergency_priority" else "Lowest simulated wait"
        cv2.putText(frame, reason, (x0 + 12, y0 + 116), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 255), 1, cv2.LINE_AA)
        return frame

    def _fixed_time_decision(self, frame_idx: int) -> SignalDecision:
        phase = SignalPhase(
            phase_id="fixed",
            label="Fixed Cycle",
            approach="north_south",
            green_seconds=30,
            estimated_wait_seconds=0.0,
            queue_pressure=0.0,
            vehicle_count=0,
        )
        return SignalDecision(frame=frame_idx, active_phase=phase, scenarios=[phase], reason="disabled")

    def _estimate_approaches(
        self,
        assignments: List[VehicleAssignment],
        lane_confidence: float,
    ) -> Dict[str, Dict[str, float]]:
        raw = {
            approach: {"pressure": 0.0, "vehicles": 0.0, "violations": 0.0}
            for approach in self.approaches
        }

        for assignment in assignments:
            approach = self._approach_for_assignment(assignment)
            weight = VEHICLE_WEIGHTS.get(assignment.cls_name, 1.0)
            if assignment.lane_id is None:
                weight *= 1.15
            if assignment.direction in ("static", "unknown"):
                weight *= 1.20
            if assignment.is_violating:
                weight *= 1.12
                raw[approach]["violations"] += 1.0

            raw[approach]["pressure"] += weight
            raw[approach]["vehicles"] += 1.0

        confidence_boost = 1.0 + max(0.0, 0.65 - lane_confidence) * 0.35
        for approach in self.approaches:
            raw[approach]["pressure"] *= confidence_boost
            previous = self.queue_history[approach]
            smoothed = {}
            for key, value in raw[approach].items():
                smoothed[key] = (1.0 - self.history_alpha) * previous[key] + self.history_alpha * value
            self.queue_history[approach] = smoothed
        return self.queue_history

    def _approach_for_assignment(self, assignment: VehicleAssignment) -> str:
        if len(self.approaches) == 1:
            return self.approaches[0]
        x, y = assignment.contact_point
        if assignment.direction == "outbound":
            return self.approaches[1]
        if assignment.direction == "inbound":
            return self.approaches[0]
        if assignment.lane_id is not None:
            return self.approaches[assignment.lane_id % len(self.approaches)]
        return self.approaches[0] if x < self.frame_w * 0.5 or y > self.frame_h * 0.55 else self.approaches[1]

    def _detect_emergency_approach(self, secondary_detections: List[Any]) -> Optional[str]:
        for det in secondary_detections:
            detector = getattr(det, "detector", "")
            label = str(getattr(det, "label", "")).lower()
            if detector == "emergency_vehicle" or "ambulance" in label:
                known = getattr(det, "approach", None)   # set when the source already knows the approach (simulator)
                if known in self.approaches:
                    return known
                bbox = getattr(det, "bbox", [0, 0, self.frame_w, self.frame_h])
                cx = (float(bbox[0]) + float(bbox[2])) / 2.0
                cy = (float(bbox[1]) + float(bbox[3])) / 2.0
                return self.approaches[0] if cx < self.frame_w * 0.5 or cy > self.frame_h * 0.55 else self.approaches[1]
        return None

    def _emergency_decision(
        self,
        frame_idx: int,
        approach_state: Dict[str, Dict[str, float]],
        approach: str,
    ) -> SignalDecision:
        pressure = approach_state.get(approach, {}).get("pressure", 0.0)
        phase = SignalPhase(
            phase_id=f"emergency_{approach}",
            label=self._label(approach),
            approach=approach,
            green_seconds=max(self.min_green, min(self.max_green, 45)),
            estimated_wait_seconds=0.0,
            queue_pressure=round(pressure, 2),
            vehicle_count=int(round(approach_state.get(approach, {}).get("vehicles", 0.0))),
            emergency_priority=True,
        )
        return SignalDecision(
            frame=frame_idx,
            active_phase=phase,
            scenarios=[phase],
            approach_state=approach_state,
            reason="emergency_priority",
        )

    def _optimize(self, frame_idx: int, approach_state: Dict[str, Dict[str, float]]) -> SignalDecision:
        scenarios: List[SignalPhase] = []
        for approach in self.approaches:
            for green in range(self.min_green, self.max_green + 1, 5):
                scenarios.append(self._simulate_phase(approach, green, approach_state))

        best = min(
            scenarios,
            key=lambda s: (
                s.estimated_wait_seconds,
                -s.queue_pressure,
                -s.vehicle_count,
                s.green_seconds,
            ),
        )
        return SignalDecision(
            frame=frame_idx,
            active_phase=best,
            scenarios=scenarios,
            approach_state=approach_state,
            reason="adaptive_min_wait",
        )

    def _simulate_phase(
        self,
        approach: str,
        green_seconds: int,
        approach_state: Dict[str, Dict[str, float]],
    ) -> SignalPhase:
        service_capacity = green_seconds * self.saturation_rate
        total_wait = 0.0
        active_pressure = approach_state[approach]["pressure"]

        for candidate_approach in self.approaches:
            state = approach_state[candidate_approach]
            pressure = state["pressure"]
            vehicles = max(1.0, state["vehicles"])
            if candidate_approach == approach:
                residual = max(0.0, pressure - service_capacity)
                wait = residual * (green_seconds + self.yellow)
            else:
                red_time = green_seconds + self.yellow + self.all_red
                spillback_penalty = max(0.0, pressure - active_pressure) * 2.5
                wait = pressure * red_time + spillback_penalty * vehicles
            wait += state["violations"] * 3.0
            total_wait += wait

        return SignalPhase(
            phase_id=f"{approach}_{green_seconds}",
            label=self._label(approach),
            approach=approach,
            green_seconds=green_seconds,
            estimated_wait_seconds=round(total_wait, 2),
            queue_pressure=round(active_pressure, 2),
            vehicle_count=int(round(approach_state[approach]["vehicles"])),
        )

    def _label(self, approach: str) -> str:
        return approach.replace("_", " ").title()
