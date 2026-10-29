"""
Number-plate recognition: plates/run_video.py (plate detector models/license_plate_best.pt + Indian-fine-tuned
reader plates/ocr/indian_reader_v2.onnx, per-character vote across frames, Indian format check).
Per vehicle it decides READ / NOT SURE / UNREADABLE. Only READ plates are ever printed (no guessed plates).

A finding is linked to a plate vehicle by position at the finding's evidence time (box overlap, or for a
rider's head: the head sitting over the vehicle box). Findings without a box (triple riding) are not linked.
"""
import json
import subprocess
from pathlib import Path

import config

ROOT = config.ROOT
TOOL = ROOT / "plates/run_video.py"
READER = ROOT / "plates/ocr/indian_reader_v2.onnx"
# the plate reader was built with the framework Python (fast-plate-ocr + onnxruntime), not the project .venv
PLATE_PY = next((p for p in ("/usr/local/bin/python3", "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3")
                 if Path(p).exists()), config.PY)
READY = TOOL.exists() and READER.exists() and (ROOT / "models/license_plate_best.pt").exists()
STATUS = "Integrated - plate detector + Indian plate reader (READ / NOT SURE / UNREADABLE)" if READY else "Model files missing"
FIELDS = ["plate", "decision", "confidence", "crop", "model", "reasons"]


def run(video: Path, work: Path, max_sec=None, env=None, tracks=None, tag=None) -> list:
    out = work / "plates" / tag if tag else work / "plates"; out.mkdir(parents=True, exist_ok=True)
    cmd = [PLATE_PY, TOOL, video, "--out", out, "--no-video"] + (["--max-sec", max_sec] if max_sec else [])
    if tracks:
        cmd += ["--tracks", ",".join(str(t) for t in tracks)]
    with open(work / f"plates{'_' + tag if tag else ''}.log", "a") as f:
        subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    j = out / f"{video.stem}_plates.json"
    if not j.exists():
        return []
    V = json.loads(j.read_text())["vehicles"]
    for v in V:
        if v.get("evidence_crop"):
            v["evidence_crop"] = str(out / v["evidence_crop"])
    return V


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def link(vehicles: list, t: float, box, head: bool, track=None) -> dict:
    """plate result for the finding's vehicle: by shared track id when known (exact), else by position at time t"""
    if not READY:
        return {"decision": "NOT AVAILABLE", "reasons": ["plate model files missing"]}
    if track is not None:
        v = next((v for v in vehicles if v["vehicle"] == track), None)
        if v is not None:
            return _result(v)
    if not box:
        return {"decision": "NOT LINKED", "reasons": ["this detector gives no vehicle box at the evidence time"]}
    best, score = None, 0.0
    for v in vehicles:
        pts = [p for p in v.get("path", []) if abs(p[0] - t) <= 0.6]
        if not pts:
            continue
        p = min(pts, key=lambda p: abs(p[0] - t))[1:]
        if head:   # rider's head: centre above / inside the vehicle box, within its width
            hx, hy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            w, h = p[2] - p[0], p[3] - p[1]
            s = 1.0 if (p[0] - 0.1 * w <= hx <= p[2] + 0.1 * w and p[1] - 1.2 * h <= hy <= p[1] + 0.6 * h) else 0.0
        else:
            s = _iou(box, p)
        if s > score:
            best, score = v, s
    if best is None or score < (0.5 if head else 0.3):
        return {"decision": "NOT LINKED", "reasons": ["no plate-tracked vehicle at this position and time"]}
    return _result(best)


def _result(best):
    out = {"decision": best["decision"], "confidence": best.get("confidence", 0.0), "reasons": best.get("reasons", []),
           "crop": best.get("evidence_crop"), "vehicle": best["vehicle"], "model": READER.name}
    out["plate"] = best.get("plate") if best["decision"] == "READ" else None     # certain readings only
    if best["decision"] == "NOT SURE":
        out["candidate"] = best.get("plate")    # unconfirmed reading: shown to the reviewer to confirm or correct
    return out


def dedupe(vehicles: list) -> list:
    """same rule as plates/run_video.py: one plate cannot belong to two vehicles - keep it only on the vehicle with
    clearly the most agreeing reads (>= 2x the next), otherwise all become NOT SURE. Applied across reader batches."""
    by = {}
    for v in vehicles:
        if v["decision"] == "READ":
            by.setdefault(v["plate"], []).append(v)
    for p, vs in by.items():
        if len(vs) < 2:
            continue
        vs.sort(key=lambda v: -v.get("reads_agreeing", 0))
        keep = vs[0].get("reads_agreeing", 0) >= 2 * vs[1].get("reads_agreeing", 0)
        for i, v in enumerate(vs):
            if i or not keep:
                v["decision"] = "NOT SURE"
                v.setdefault("reasons", []).append(f"same plate also read on {len(vs) - 1} other vehicle(s) - cannot be sure which vehicle it belongs to")
    return vehicles
