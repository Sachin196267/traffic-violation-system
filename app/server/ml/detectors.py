"""
Adapters to the trained tools in this project. Each tool runs as a subprocess (its own models, tracking and
multi-frame decision rules, thresholds tuned on held-out cameras) and its JSON output is mapped to Findings.

  helmet     helmet_training/robust/run_video.py     runs/helmet_robust/final        (IDD detector + COCO check + head classifier, per-rider vote over frames)
  triple     triple_training/run_video.py            runs/triple_rider/final         (rider-count classifier + heads-on-seat, frame agreement)
  redlight   red_light/observe.py + decide.py        runs/signal_state + runs/ambulance_cls  (signal state HMM, stop-line crossing; automatic - stop line found from the red-light queue)
  speed      speed_estimation/run.py                 calibrated range + confidence (needs a calibration file)
  lanes      lane_trace/lane_trace.py                context only - never creates a finding
  emergency  CLIP probe runs/ambulance_cls           applied inside redlight/speed: ambulance / police / fire are exempt
  plate      NOT IMPLEMENTED - model still in training (see plate.py for the integration boundary)
"""
import json
import subprocess
from pathlib import Path

import config
from ml.base import Context, Detector, Finding

ROOT = config.ROOT


def _run(cmd, log: Path, cwd=ROOT, env=None):
    with open(log, "a") as f:
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return r.returncode == 0


def _op(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else {}


class HelmetDetector(Detector):
    key, title = "helmet", "Helmet check"
    produces = ["Riding without helmet"]
    weights = ROOT / "runs/helmet_robust/final/weights/last.pt"

    def available(self):
        return (self.weights.exists(), "ready" if self.weights.exists() else "weights missing")

    def run(self, video, work, opts, progress):
        out = work / "helmet"; out.mkdir(exist_ok=True)
        args = [config.PY, ROOT / "helmet_training/robust/run_video.py", video, "--out", out]
        if opts.get("max_sec"):
            args += ["--max-sec", opts["max_sec"]]
        if not _run(args, work / "helmet.log", env=opts.get("_shared_env") or opts.get("_env")):
            raise RuntimeError("helmet tool failed (see helmet.log)")
        J = json.loads((out / f"{video.stem}_riders.json").read_text())
        op = J.get("operating_point", {})
        info = {"model": "runs/helmet_robust/final (YOLO11 head classifier)", "operating_point": op,
                "rule": "per-rider size-weighted vote over all frames; not sure between thresholds"}
        F = []
        for r in J["riders"]:
            if r["decision"] == "helmet":
                continue
            p = r["p"]
            route = "confirmed" if r["decision"] == "no_helmet" else "review"
            unc = []
            if r["decision"] == "not_sure":
                unc.append("the helmet model's vote is between its 'helmet' and 'no helmet' thresholds")
            if route == "confirmed" and r["slot"] >= 1:
                # known failure mode: a person standing beside the bike counted as a rider (3 of 22 checked
                # challans). Extra persons on a bike therefore always get a human look.
                route = "review"
                unc.append("second/extra person on this motorcycle - could be a pillion rider or a bystander next to the bike")
            if r["frames"] < 3:
                route = "review"; unc.append(f"seen in only {r['frames']} frames")
            F.append(Finding(
                detector=self.key, violation="Riding without helmet", track=f"H{r['track']}.{r['slot']}",
                t=r["t_best"], t_start=max(0, r["t_best"] - 2), t_end=r["t_best"] + 2,
                confidence=round(p.get("no_helmet", 0), 3), model_decision=r["decision"], route=route,
                reasons=[f"no-helmet probability {p.get('no_helmet', 0):.2f} (helmet {p.get('helmet', 0):.2f}), vote over {r['frames']} frames",
                         f"clearest head view at {r['t_best']:.1f} s, {r['max_head_px']:.0f} px tall",
                         f"rider slot {r['slot'] + 1} on motorcycle track {r['track']}"],
                uncertain_reasons=unc, box=r.get("head_box_best"), vehicle="motorcycle", model_info=info))
        arts = {"annotated": out / f"{video.stem}_helmet.mp4", "json": out / f"{video.stem}_riders.json"}
        return F, arts


class TripleDetector(Detector):
    key, title = "triple", "Triple-riding check"
    produces = ["Triple riding"]
    weights = ROOT / "runs/triple_rider/final/weights/last.pt"

    def available(self):
        ok = self.weights.exists() and (ROOT / "runs/triple_rider/operating_point.json").exists()
        return ok, "ready" if ok else "weights or operating point missing"

    def run(self, video, work, opts, progress):
        out = work / "triple"; out.mkdir(exist_ok=True)
        args = [config.PY, ROOT / "triple_training/run_video.py", video, "--out", out]
        if opts.get("max_sec"):
            args += ["--max-sec", opts["max_sec"]]
        if not _run(args, work / "triple.log", env=opts.get("_shared_env") or opts.get("_env")):
            raise RuntimeError("triple-riding tool failed (see triple.log)")
        J = json.loads((out / f"{video.stem}_triple.json").read_text())
        F = []
        for r in J["motorcycles"]:
            if r["decision"] == "not_triple":
                continue
            route = "confirmed" if r["decision"] == "triple" else "review"
            unc = [r["reasons"][-1]] if route == "review" else []
            F.append(Finding(
                detector=self.key, violation="Triple riding", track=f"T{r['track']}",
                t=r["first_seen_s"], t_start=r["first_seen_s"], t_end=r["first_seen_s"] + 3,
                confidence=r["confidence"], model_decision=r["decision"], route=route,
                reasons=r["reasons"], uncertain_reasons=unc,
                box=None,   # the tool reports the LAST box, not the box at first_seen - not drawn on evidence
                vehicle="motorcycle",
                model_info={"model": "runs/triple_rider/final (YOLO11 rider-count classifier + heads on seat)",
                            "thresholds": r.get("thresholds")}))
        return F, {"annotated": out / f"{video.stem}_triple.mp4", "json": out / f"{video.stem}_triple.json"}


class RedLightDetector(Detector):
    key, title = "redlight", "Red-light check"
    produces = ["Red-light jumping"]
    needs = None                  # automatic: finds the signal, direction and stop line itself

    def available(self):
        ok = (ROOT / "runs/signal_state/weights/last.pt").exists()
        return ok, "ready (automatic on every video, no setup)" if ok else "signal model missing"

    def run(self, video, work, opts, progress):
        runs = work / "redlight"; runs.mkdir(exist_ok=True)
        args = [config.PY, ROOT / "red_light/observe.py", video, "--out", runs]
        if opts.get("max_sec"):
            args += ["--max-sec", opts["max_sec"]]
        if not _run(args, work / "redlight.log", env=opts.get("_env")):
            raise RuntimeError("red-light observe failed (see redlight.log)")
        d = [config.PY, ROOT / "red_light/decide.py", video, "--runs", runs, "--render"]
        if opts.get("camera"):
            d += ["--camera", opts["camera"]]
        if not _run(d, work / "redlight.log", env=opts.get("_env")):
            raise RuntimeError("red-light decide failed (see redlight.log)")
        J = json.loads((runs / video.stem / "decisions.json").read_text())
        F = []
        for r in J["decisions"]:
            if r["decision"] == "OK":
                continue
            route = {"VIOLATION": "confirmed", "NOT SURE": "review", "EXEMPT": "exempt"}[r["decision"]]
            unc = r["reasons"][2:] if route == "review" else []
            F.append(Finding(
                detector=self.key, violation="Red-light jumping", track=f"R{r['track']}",
                t=r["t_cross"], t_start=max(0, r["t_cross"] - 3), t_end=r["t_cross"] + 2,
                confidence=r["confidence"], model_decision=r["decision"], route=route,
                reasons=r["reasons"], uncertain_reasons=unc, box=r["box"], vehicle=r["class"],
                emergency=r.get("emergency_type") if route == "exempt" else None,
                model_info={"signal_model": "runs/signal_state (YOLO11 red/yellow/green/not-light + HMM)",
                            "emergency_model": J["summary"].get("ambulance_model"),
                            "stop_line": r.get("stop_line_source")}))
        notes = [f"red light {s['t'][0]:.0f}-{s['t'][1]:.0f} s: " + ("no traffic signal visible - nothing to check"
                 if s["reasons"][:1] and s["reasons"][0].startswith("no signal head") else "not checked - " + "; ".join(s["reasons"]))
                 for s in J["summary"]["shots"] if not s.get("ready")]
        return F, {"annotated": runs / video.stem / f"{video.stem}_redlight.mp4", "json": runs / video.stem / "decisions.json",
                   "_notes": notes}


class SpeedDetector(Detector):
    key, title = "speed", "Speed estimate"
    produces = ["Over-speeding"]
    needs = "calibration file (one known road length) + speed limit"

    def run(self, video, work, opts, progress):
        if not (opts.get("speed_calib") and opts.get("speed_limit")):
            return [], {"_notes": ["speed not checked: needs a calibration file and a speed limit"]}
        args = [config.PY, ROOT / "speed_estimation/run.py", video, "--calib", opts["speed_calib"]]
        if opts.get("max_sec"):
            args += ["--max-sec", opts["max_sec"]]
        if not _run(args, work / "speed.log", env=opts.get("_env")):
            raise RuntimeError("speed tool failed (see speed.log)")
        sd = ROOT / "speed_estimation/runs" / video.stem
        S = json.loads((sd / "speeds.json").read_text())["vehicles"]
        T = json.loads((sd / "tracks.json").read_text())["tracks"]
        lim = float(opts["speed_limit"]); F = []
        for tid, v in S.items():
            if "range90_kmh" not in v or v["confidence"] not in ("high", "medium", "low") or v["median_kmh"] <= lim:
                continue
            lo, hi = v["range90_kmh"]
            route = "confirmed" if lo > lim and v["confidence"] in ("high", "medium") else "review"
            tr = T[tid]; k = len(tr["t"]) // 2
            F.append(Finding(
                detector=self.key, violation="Over-speeding", track=f"S{tid}", t=tr["t"][k], t_start=tr["t"][0], t_end=tr["t"][-1],
                confidence=round(max(0.0, 1 - 0.5 * (hi - lo) / max(v["median_kmh"], 1)), 3), model_decision=v["confidence"], route=route,
                reasons=[f"speed {lo:.0f}-{hi:.0f} km/h (90 % range, {v['confidence']} confidence), limit {lim:.0f} km/h"],
                uncertain_reasons=[] if route == "confirmed" else ["the 90 % range includes speeds within the limit"],
                box=tr.get("box_img", tr["box"])[k], vehicle=v.get("cls", ""),
                model_info={"model": "speed_estimation (calibrated camera geometry, Monte-Carlo range)"}))
        return F, {"annotated": sd / f"{video.stem}_speed.mp4", "json": sd / "speeds.json"}


class LaneContext:
    """LaneTrace: lanes estimated from vehicle trajectories (paint ignored). Context only - never a finding."""
    key, title = "lanes", "Lane structure (context only)"

    def run(self, video, work, opts, progress) -> Context:
        out = work / "lanes"
        ok = _run([config.PY, ROOT / "lane_trace/lane_trace.py", "--video", video, "--seconds", min(30, float(opts.get("max_sec") or 30)),
                   "--out", out], work / "lanes.log", env=opts.get("_env"))
        rep = json.loads((out / "report.json").read_text()) if ok and (out / "report.json").exists() else {}
        files = {k: out / f for k, f in (("overlay", "lanes_preview.jpg"), ("background_lanes", "lanes_background.jpg"), ("video", "lanes_video.mp4"), ("lanes", "lanes.json")) if (out / f).exists()}
        return Context("lanes", files, {"ok": bool(files), "n_lanes": (rep.get("final") or {}).get("n_lanes") if isinstance(rep.get("final"), dict) else None,
                                        "note": "Lanes are estimated from how traffic moves. They support positioning and the digital twin only - "
                                                "lane position never creates a challan on its own."})


REGISTRY = {d.key: d for d in (HelmetDetector(), TripleDetector(), RedLightDetector(), SpeedDetector())}
LANES = LaneContext()
