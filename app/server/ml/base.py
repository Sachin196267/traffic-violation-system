"""
Model layer interface. Every analysis module is a Detector: it runs on the whole video (multi-frame, with tracking),
writes its own artefacts into a work folder and returns Findings. The pipeline never looks inside a model.

To add or replace a model: write a Detector subclass, register it in detectors.REGISTRY. Nothing else changes.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Finding:
    detector: str                 # registry key of the detector that produced it
    violation: str                # human label, e.g. "Riding without helmet"
    track: str                    # tracking id within this analysis
    t: float                      # best evidence time (s)
    t_start: float
    t_end: float
    confidence: float             # 0..1, as reported by the model's own decision rule
    model_decision: str           # the model's raw decision ("no_helmet", "triple", "VIOLATION", ...)
    route: str                    # "confirmed" (auto challan) | "review" (human) | "exempt" (emergency vehicle)
    reasons: list = field(default_factory=list)            # plain-language evidence, from the model output
    uncertain_reasons: list = field(default_factory=list)  # why it went to human review
    box: list | None = None       # [x1,y1,x2,y2] at time t, video pixels (None when the model gives no box)
    vehicle: str = ""
    emergency: str | None = None
    model_info: dict = field(default_factory=dict)          # weights, thresholds, operating point


@dataclass
class Context:
    """Supporting output that never creates a finding by itself (e.g. lanes)."""
    kind: str
    files: dict                   # name -> path
    summary: dict


class Detector:
    key = "base"
    title = "Detector"
    produces = []                 # violation labels it can produce
    needs = None                  # None, or a string describing a required per-camera setup

    def available(self) -> tuple[bool, str]:
        return True, "ready"

    def run(self, video: Path, work: Path, opts: dict, progress) -> tuple[list[Finding], dict]:
        """returns findings and a dict of artefacts {name: path} (annotated video, json, ...)"""
        raise NotImplementedError
