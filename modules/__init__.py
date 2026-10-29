"""
Robust Automatic Lane Detection & Assignment Modules
for Unstructured Indian Traffic Surveillance
"""

from .tracker_pipeline import TrackerPipeline, TrackObservation, TrackSummary
from .perspective_transformer import PerspectiveTransformer
from .scene_analyzer import SceneAnalyzer, RoadCorridor
from .trajectory_clustering import TrajectoryClusterer, InferredLane
from .temporal_lane_model import TemporalLaneModel, PersistentBoundary
from .lane_assigner import LaneAssigner, VehicleAssignment
from .visualizer import LaneVisualizer
from .smart_signal_controller import SmartSignalController, SignalDecision, SignalPhase
from .xai_explainer import ViolationXAIExplainer, XAIExplanation
from .violation_slip_generator import ViolationSlipGenerator, SlipResult
from .anpr_reader import LicensePlateReader
from .temporal_violation_analyzer import TemporalViolationAnalyzer, PlateCandidate
from .severity_engine import SeverityEngine, SeverityAssessment

__all__ = [
    "TrackerPipeline",
    "TrackObservation",
    "TrackSummary",
    "PerspectiveTransformer",
    "SceneAnalyzer",
    "RoadCorridor",
    "TrajectoryClusterer",
    "InferredLane",
    "TemporalLaneModel",
    "PersistentBoundary",
    "LaneAssigner",
    "VehicleAssignment",
    "LaneVisualizer",
    "SmartSignalController",
    "SignalDecision",
    "SignalPhase",
    "ViolationXAIExplainer",
    "XAIExplanation",
    "ViolationSlipGenerator",
    "SlipResult",
    "LicensePlateReader",
    "TemporalViolationAnalyzer",
    "PlateCandidate",
    "SeverityEngine",
    "SeverityAssessment",
]
