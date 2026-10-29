# Loaded (via PYTHONPATH) into every tool process started by the platform. Two independent switches:
#
#  TVP_DEVICE=mps        run every YOLO model on the Mac GPU. Ultralytics only uses CUDA automatically and
#                        otherwise falls back to the CPU, so without this the GPU stays idle.
#  TVP_DET_CACHE=<pkl>   `YOLO(".../runs/idd_final/weights/best.pt")` returns a replay of the shared detection
#                        pass (ml/shared_detect.py) instead of loading and running the detector again. Each .track()
#                        call returns the next cached frame - the tools call it once per sampled frame, in the
#                        same order as the pass, so detections and track IDs are identical.
import os, sys

_DEV = os.environ.get("TVP_DEVICE")
_CACHE = os.environ.get("TVP_DET_CACHE")
if _CACHE and not os.path.exists(_CACHE):
    _CACHE = None

if _DEV or _CACHE:
    import importlib.abc, importlib.util, pickle

    _IDD = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../runs/idd_final/weights/best.pt"))

    class _Boxes:
        def __init__(self, d):
            import torch
            self.xyxy = torch.from_numpy(d["xyxy"]); self.cls = torch.from_numpy(d["cls"]); self.conf = torch.from_numpy(d["conf"])
            self.id = None if d["id"] is None else torch.from_numpy(d["id"])

        def __len__(self):
            return len(self.xyxy)

    class _Result:
        def __init__(self, d, names):
            self.boxes = _Boxes(d); self.names = names

    class _Replay:
        def __init__(self, path):
            with open(path, "rb") as fh:
                c = pickle.load(fh)
            self.names = c["names"]; self._frames = c["frames"]; self._i = 0

        def track(self, *a, **k):
            if self._i >= len(self._frames):
                raise RuntimeError("detection cache exhausted - tool sampled more frames than the shared pass")
            d = self._frames[self._i]; self._i += 1
            return [_Result(d, self.names)]

    def _patch(m):
        if _DEV:
            from ultralytics.engine.model import Model
            pred = Model.predict

            def predict(self, *a, **k):
                k.setdefault("device", _DEV)
                return pred(self, *a, **k)
            Model.predict = predict            # track() and __call__ go through predict()
        if _CACHE:
            Y = m.YOLO; orig_new = Y.__new__

            def new(cls, model="yolo11n.pt", *a, **k):
                if os.path.abspath(str(model)) == _IDD:
                    return _Replay(_CACHE)
                return orig_new(cls) if orig_new is object.__new__ else orig_new(cls, model, *a, **k)
            Y.__new__ = new

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "ultralytics":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name); run = spec.loader.exec_module

            def exec_module(m):
                run(m)
                _patch(m)
            spec.loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, _Hook())
