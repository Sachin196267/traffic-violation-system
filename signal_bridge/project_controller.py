"""
Drives the project's own adaptive controller (modules/smart_signal_controller.py) from the simulator.

The simulator's junction cameras report the vehicles waiting on each approach; they are handed to
SmartSignalController.update() as VehicleAssignments, exactly as the video pipeline would (lane_id = approach),
and its decision (approach + green seconds) runs the signal.
"""
import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["north", "east", "south", "west"]
CLS = {"Bike": "motorcycle", "Bicycle": "bicycle", "Auto": "autorickshaw", "Car": "car", "Bus": "bus", "Truck": "truck", "Ambulance": "car"}
VIDEO_FPS = 30   # the controller counts frames; the simulator time is mapped onto a 30 fps video clock


def _load():
    """import modules.smart_signal_controller without modules/__init__.py (which pulls in every pipeline module)"""
    if "modules" not in sys.modules:
        pkg = types.ModuleType("modules")
        pkg.__path__ = [str(ROOT / "modules")]
        sys.modules["modules"] = pkg
    return importlib.import_module("modules.smart_signal_controller"), importlib.import_module("modules.lane_assigner")


def _config():
    cfg = {}
    try:
        import yaml
        cfg = (yaml.safe_load((ROOT / "config/integrated_system_config.yaml").read_text()) or {}).get("smart_signal", {})
    except Exception:
        pass
    cfg = dict(cfg)
    cfg["approaches"] = NAMES          # the simulated junction has four approaches
    cfg["enabled"] = True
    return {"smart_signal": cfg}


class _Emergency:
    def __init__(self, approach):
        self.detector, self.label, self.approach, self.bbox = "emergency_vehicle", "ambulance", approach, [0, 0, 1, 1]


class ProjectController:
    name = "Project SmartSignalController"

    def __init__(self):
        ssc, la = _load()
        self.VA = la.VehicleAssignment
        self.ctl = ssc.SmartSignalController(_config(), 1920, 1080)
        self.min_green = self.ctl.min_green
        self.last = None

    def _decide(self, m):
        assignments = [
            self.VA(track_id=v["id"], cls_name=CLS.get(v["type"], "car"), lane_id=v["a"], lane_name=NAMES[v["a"]], lane_conf=1.0,
                    dist_from_center_px=0.0, direction="static" if v["v"] < 0.5 else "approaching", contact_point=(0.0, 0.0),
                    speed_kmh=v["v"] * 3.6, is_violating=bool(v["jumper"]))
            for v in m["veh"]
        ]
        amb = [_Emergency(NAMES[a]) for a in range(4) if m["app"][a]["amb"]]
        d = self.ctl.update(int(round(m["t"] * VIDEO_FPS)), assignments, amb, 1.0)
        self.last = d
        return NAMES.index(d.active_phase.approach), float(d.active_phase.green_seconds), d

    @staticmethod
    def _why(d):
        p = d.active_phase
        if d.reason == "emergency_priority":
            return f"Project controller: emergency priority for {p.label}"
        return f"Project controller: {p.label} {p.green_seconds} s - lowest projected wait {p.estimated_wait_seconds:.0f} (pressure {p.queue_pressure:.1f})"

    def next(self, m):
        a, g, d = self._decide(m)
        return a, g, self._why(d)

    def hold(self, m):
        a, g, d = self._decide(m)
        cur, el = m["green"], m["phase_t"]
        if a == cur:
            if d.reason == "emergency_priority" or el < max(g, self.min_green):
                return True, ""
            return False, f"Project controller: planned {g:.0f} s green over"
        if el < self.min_green:
            return True, ""
        return False, f"Project controller now prefers {d.active_phase.label}"
