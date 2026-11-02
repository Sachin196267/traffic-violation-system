"""
modules/severity_engine.py

Configurable Severity and Intensity Assessment Engine for Traffic Violations.
Calculates numeric hazard scores (0-100), maps them to severity tiers
(LOW, MEDIUM, HIGH, CRITICAL), and generates transparent human-readable
reasoning explaining each contributing risk factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SeverityAssessment:
    score: int
    level: str  # LOW, MEDIUM, HIGH, CRITICAL
    badge_color: str
    reasons: List[str] = field(default_factory=list)
    factors: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "level": self.level,
            "badge_color": self.badge_color,
            "reasons": self.reasons,
            "factors": self.factors,
        }


class SeverityEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        sev_cfg = self.config.get("severity", {})
        self.enabled = sev_cfg.get("enabled", True)

        self.levels = sev_cfg.get(
            "levels",
            {
                "LOW": {"min_score": 0, "max_score": 35, "badge_color": "#10b981"},
                "MEDIUM": {"min_score": 36, "max_score": 65, "badge_color": "#f59e0b"},
                "HIGH": {"min_score": 66, "max_score": 85, "badge_color": "#f97316"},
                "CRITICAL": {"min_score": 86, "max_score": 100, "badge_color": "#ef4444"},
            },
        )

        self.base_scores = sev_cfg.get(
            "base_scores",
            {
                "EMERGENCY_OBSTRUCTION": 90,
                "WRONG_WAY_DRIVING": 85,
                "RED_LIGHT_VIOLATION": 75,
                "TRIPLE_RIDING": 65,
                "DANGEROUS_WEAVING": 60,
                "NO_HELMET": 55,
                "SPEEDING": 50,
                "STOP_LINE_ENCROACHMENT": 30,
            },
        )

        self.modifiers = sev_cfg.get(
            "modifiers",
            {
                "speed_excess_factor": 1.0,
                "heavy_vehicle_bonus": 15,
                "compound_violation_bonus": 15,
                "unprotected_two_wheeler_compound_bonus": 10,
            },
        )

        assign_cfg = self.config.get("lane_assignment", {})
        self.speed_limit_kmh = float(assign_cfg.get("speed_limit_kmh", 65.0))

    def evaluate(
        self,
        violation_types: List[str],
        cls_name: str = "",
        speed_kmh: float = 0.0,
        lane_confidence: float = 1.0,
        emergency_present: bool = False,
    ) -> SeverityAssessment:
        if not violation_types:
            return SeverityAssessment(
                score=0,
                level="LOW",
                badge_color=self.levels.get("LOW", {}).get("badge_color", "#10b981"),
                reasons=["No violation detected."],
                factors={"base_violation_max": 0},
            )

        reasons: List[str] = []
        factors: Dict[str, Any] = {}

        # 1. Base Score: Take maximum base score among triggered violations
        base_score = 0
        primary_violation = ""
        for v in violation_types:
            score = self.base_scores.get(v, 40)
            if score > base_score:
                base_score = score
                primary_violation = v

        reasons.append(f"Primary violation '{primary_violation}' establishes baseline severity of {base_score}/100.")
        factors["primary_violation"] = primary_violation
        factors["base_score"] = base_score

        current_score = float(base_score)

        # 2. Compound Violation Penalty: Each additional violation compounds danger
        num_violations = len(violation_types)
        if num_violations > 1:
            compound_bonus = float(self.modifiers.get("compound_violation_bonus", 15)) * (num_violations - 1)
            current_score += compound_bonus
            other_violations = [v for v in violation_types if v != primary_violation]
            reasons.append(
                f"Multiple concurrent violations ({', '.join(other_violations)}) add +{compound_bonus:.0f} points."
            )
            factors["compound_violations_count"] = num_violations
            factors["compound_bonus"] = compound_bonus

        # 3. Speed Excess Modifier
        if "SPEEDING" in violation_types or speed_kmh > self.speed_limit_kmh:
            excess_speed = max(0.0, speed_kmh - self.speed_limit_kmh)
            if excess_speed > 0:
                speed_factor = float(self.modifiers.get("speed_excess_factor", 1.0))
                speed_penalty = min(25.0, excess_speed * speed_factor)
                current_score += speed_penalty
                reasons.append(
                    f"Measured velocity of {speed_kmh:.1f} km/h is {excess_speed:.1f} km/h over the {self.speed_limit_kmh:.0f} km/h limit (+{speed_penalty:.0f} points)."
                )
                factors["excess_speed_kmh"] = round(excess_speed, 1)
                factors["speed_penalty"] = round(speed_penalty, 1)

        # 4. Vehicle Mass Hazard (Heavy vehicles: bus, truck, trailer)
        is_heavy = cls_name.lower() in ("bus", "truck", "trailer", "caravan")
        if is_heavy:
            heavy_bonus = float(self.modifiers.get("heavy_vehicle_bonus", 15))
            current_score += heavy_bonus
            reasons.append(
                f"High-mass vehicle category ({cls_name.upper()}) poses heightened kinetic impact risk (+{heavy_bonus:.0f} points)."
            )
            factors["heavy_vehicle"] = True
            factors["heavy_vehicle_bonus"] = heavy_bonus

        # 5. Compound Unprotected Two-Wheeler Vulnerability
        is_two_wheeler = cls_name.lower() in ("motorcycle", "bicycle")
        has_no_helmet = "NO_HELMET" in violation_types
        has_triple_riding = "TRIPLE_RIDING" in violation_types
        if is_two_wheeler and has_no_helmet and has_triple_riding:
            unprotected_bonus = float(self.modifiers.get("unprotected_two_wheeler_compound_bonus", 10))
            current_score += unprotected_bonus
            reasons.append(
                f"Concurrent triple riding with unhelmeted occupants presents acute fatality risk (+{unprotected_bonus:.0f} points)."
            )
            factors["unprotected_two_wheeler_compound"] = True
            factors["unprotected_bonus"] = unprotected_bonus

        # 6. Clamp score to 0..100
        final_score = int(round(max(0.0, min(100.0, current_score))))

        # 7. Map score to level
        level = "LOW"
        badge_color = "#10b981"
        for lvl_name, lvl_cfg in self.levels.items():
            min_s = lvl_cfg.get("min_score", 0)
            max_s = lvl_cfg.get("max_score", 100)
            if min_s <= final_score <= max_s:
                level = lvl_name
                badge_color = lvl_cfg.get("badge_color", "#10b981")
                break

        reasons.append(f"Aggregate risk score evaluates to {final_score}/100, classified as {level} severity.")
        factors["final_score"] = final_score
        factors["level"] = level

        return SeverityAssessment(
            score=final_score,
            level=level,
            badge_color=badge_color,
            reasons=reasons,
            factors=factors,
        )
