#!/usr/bin/env python3
"""
Standalone e-challan tool: runs every violation check on a video and lists who gets a challan.
Not connected to the website (user rule).

    python echallan/run.py <video> [--checks helmet,triple,redlight,speed] [--camera cam.json]
                                   [--speed-calib calib.json --speed-limit 50] [--max-sec 120]

Checks (each is its own standalone tool, run as a subprocess; results cached in echallan/runs/<video>/):
  helmet    helmet_training/robust/run_video.py    rider without helmet
  triple    triple_training/run_video.py           3+ people on a two-wheeler
  redlight  red_light/observe.py + decide.py       crossing the stop line on red
  speed     speed_estimation/run.py                only with a calibration file and a speed limit: challan when
                                                   even the LOW end of the 90 % speed range is over the limit
Every finding becomes one of
  CHALLAN  the tool is sure (its own precision-first thresholds, tuned on held-out cameras)
  REVIEW   the tool is not sure - a person must look before any challan
  EXEMPT   emergency vehicle (ambulance / police / fire, CLIP probe, runs/ambulance_cls) - never challaned
with a confidence, plain-language reasons and an evidence image (frame + box).
Writes echallan/runs/<video>/challans.json, challans.md and evidence/*.jpg.
"""
import argparse, json, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--checks", default="helmet,triple,redlight,speed")
ap.add_argument("--camera", help="red-light camera config (stop line / direction / light)")
ap.add_argument("--speed-calib")
ap.add_argument("--speed-limit", type=float)
ap.add_argument("--max-sec", type=float, default=120)
ap.add_argument("--rerun", action="store_true", help="ignore cached tool outputs")
args = ap.parse_args()

vp = Path(args.video).absolute()      # keep a symlink's own name (separate run folders)
out = ROOT / "echallan/runs" / vp.stem
(out / "evidence").mkdir(parents=True, exist_ok=True)
checks = [c.strip() for c in args.checks.split(",") if c.strip()]
log = open(out / "tools.log", "a")
notes = []


def run(cmd, marker):
    if Path(marker).exists() and not args.rerun:
        return True
    print("running:", " ".join(map(str, cmd[1:3])), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return r.returncode == 0 and Path(marker).exists()


cap = cv2.VideoCapture(str(vp))


def evidence(name, t, box, colour, text):
    """frame at time t with the box drawn -> evidence/<name>.jpg"""
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    if not ok:
        return None
    if box:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(fr, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(fr, text, (x1, max(20, y1 - 8)), 0, 0.7, colour, 2)
    p = out / "evidence" / f"{name}.jpg"
    cv2.imwrite(str(p), fr)
    return str(p.relative_to(ROOT))


items = []
RED, ORANGE, BLUE = (0, 0, 255), (0, 170, 255), (255, 160, 0)
ms = ["--max-sec", str(args.max_sec)] if args.max_sec else []

# ---------------- helmet ----------------
if "helmet" in checks:
    d = out / "helmet"
    if run([PY, ROOT / "helmet_training/robust/run_video.py", vp, "--out", d] + ms, d / f"{vp.stem}_riders.json"):
        J = json.loads((d / f"{vp.stem}_riders.json").read_text())
        for r in J["riders"]:
            if r["decision"] == "helmet":
                continue
            p = r["p"]
            dec = "CHALLAN" if r["decision"] == "no_helmet" else "REVIEW"
            reasons = [f"no-helmet probability {p.get('no_helmet', 0):.2f} (helmet {p.get('helmet', 0):.2f}) over {r['frames']} frames",
                       f"clearest head seen at {r['t_best']} s, {r['max_head_px']:.0f} px tall"]
            if dec == "REVIEW":
                reasons.append("the helmet model is not sure (between its thresholds) - check the evidence image")
            box = r.get("head_box_best")
            items.append({"violation": "riding without helmet", "decision": dec, "confidence": round(p.get("no_helmet", 0), 3),
                          "vehicle": "motorcycle", "t": r["t_best"], "reasons": reasons, "box": box,
                          "evidence": evidence(f"helmet_{r['track']}_{r['slot']}", r["t_best"], box, RED if dec == "CHALLAN" else ORANGE,
                                               "NO HELMET" if dec == "CHALLAN" else "helmet?")})
    else:
        notes.append("helmet check failed - see tools.log")

# ---------------- triple riding ----------------
if "triple" in checks:
    d = out / "triple"
    if run([PY, ROOT / "triple_training/run_video.py", vp, "--out", d] + ms, d / f"{vp.stem}_triple.json"):
        J = json.loads((d / f"{vp.stem}_triple.json").read_text())
        for r in J["motorcycles"]:
            if r["decision"] == "not_triple":
                continue
            dec = "CHALLAN" if r["decision"] == "triple" else "REVIEW"
            items.append({"violation": "triple riding", "decision": dec, "confidence": r["confidence"], "vehicle": "motorcycle",
                          "t": r["first_seen_s"], "reasons": r["reasons"], "box": r.get("last_box"),
                          "evidence": evidence(f"triple_{r['track']}", r["first_seen_s"], r.get("last_box"),
                                               RED if dec == "CHALLAN" else ORANGE, "TRIPLE" if dec == "CHALLAN" else "triple?")})
    else:
        notes.append("triple-riding check failed - see tools.log")

# ---------------- red light ----------------
if "redlight" in checks:
    runs = out / "redlight"
    ok = run([PY, ROOT / "red_light/observe.py", vp, "--out", runs] + ms, runs / vp.stem / "obs.json")
    cam = ["--camera", args.camera] if args.camera else []
    ok = ok and run([PY, ROOT / "red_light/decide.py", vp, "--runs", runs] + cam, runs / vp.stem / "decisions.json")
    if ok:
        J = json.loads((runs / vp.stem / "decisions.json").read_text())
        for s in J["summary"]["shots"]:
            if not s.get("ready"):
                notes.append(f"red light, shot {s['t'][0]:.0f}-{s['t'][1]:.0f} s not checked: " + "; ".join(s["reasons"]))
        for r in J["decisions"]:
            if r["decision"] == "OK":
                continue
            dec = {"VIOLATION": "CHALLAN", "NOT SURE": "REVIEW", "EXEMPT": "EXEMPT"}[r["decision"]]
            col = {"CHALLAN": RED, "REVIEW": ORANGE, "EXEMPT": BLUE}[dec]
            items.append({"violation": "red-light jumping", "decision": dec, "confidence": r["confidence"], "vehicle": r["class"],
                          "t": r["t_cross"], "reasons": r["reasons"], "emergency": r.get("emergency_type") if dec == "EXEMPT" else None,
                          "box": r["box"],
                          "evidence": evidence(f"redlight_{r['track'].replace(':', '_')}", r["t_cross"], r["box"], col,
                                               {"CHALLAN": "RED LIGHT", "REVIEW": "red light?", "EXEMPT": "EMERGENCY - exempt"}[dec])})
    else:
        notes.append("red-light check failed - see tools.log")

# ---------------- speed ----------------
if "speed" in checks:
    if not (args.speed_calib and args.speed_limit):
        notes.append("speed not checked: needs --speed-calib (one known road length) and --speed-limit; "
                     "uncalibrated speeds are too uncertain for challans")
    else:
        sd = ROOT / "speed_estimation/runs"          # the speed tool keeps its own run folder
        ok = run([PY, ROOT / "speed_estimation/run.py", vp, "--calib", args.speed_calib] + ms, sd / vp.stem / "speeds.json")
        if ok:
            S = json.loads((sd / vp.stem / "speeds.json").read_text())["vehicles"]
            T = json.loads((sd / vp.stem / "tracks.json").read_text())["tracks"]
            sys.path.insert(0, str(ROOT / "ambulance_training"))
            probe = None
            for tid, v in S.items():
                if "range90_kmh" not in v or v["confidence"] not in ("high", "medium", "low"):
                    continue
                lo, med, hi = v["range90_kmh"][0], v["median_kmh"], v["range90_kmh"][1]
                if med <= args.speed_limit:
                    continue
                dec = "CHALLAN" if lo > args.speed_limit and v["confidence"] in ("high", "medium") else "REVIEW"
                tr = T[tid]; k = len(tr["t"]) // 2
                t, box = tr["t"][k], tr.get("box_img", tr["box"])[k]
                reasons = [f"speed {lo:.0f}-{hi:.0f} km/h (90 % range, {v['confidence']} confidence), limit {args.speed_limit:.0f} km/h"]
                reasons.append("even the lowest likely speed is over the limit" if dec == "CHALLAN"
                               else "the range includes speeds within the limit - not certain enough for a challan")
                items.append({"violation": "overspeeding", "decision": dec, "confidence": round(1 - 0.5 * (hi - lo) / max(med, 1), 3),
                              "vehicle": v["cls"], "t": t, "box": box, "reasons": reasons, "_speed_track": tid})
        else:
            notes.append("speed check failed - see tools.log")

# ---------------- emergency-vehicle exemption for four-wheeler findings not already checked ----------------
need = [it for it in items if it.get("_speed_track")]
if need:
    from clip_features import EmergencyProbe
    op = json.loads((ROOT / "runs/ambulance_cls/operating_point.json").read_text())
    ep = EmergencyProbe(ROOT / "runs/ambulance_cls/probe.npz")
    em = [ep.classes.index(c) for c in ("ambulance", "police", "fire")]
    for it in need:
        cap.set(cv2.CAP_PROP_POS_MSEC, it["t"] * 1000); ok, fr = cap.read()
        x1, y1, x2, y2 = map(int, it["box"])
        P = ep.predict([fr[max(0, y1):y2, max(0, x1):x2]]) if ok else None
        s = float(P[0][em].sum()) if P is not None else 0.0
        ty = ep.classes[em[int(np.argmax(P[0][em]))]] if P is not None else None
        if s >= op["t_exempt"]:
            it["decision"] = "EXEMPT"; it["emergency"] = ty
            it["reasons"].append(f"emergency vehicle: {ty} (score {s:.2f}) - emergency vehicles are not challaned")
        elif s >= op["t_review"] and it["decision"] == "CHALLAN":
            it["decision"] = "REVIEW"; it["reasons"].append(f"may be an emergency vehicle ({ty}, score {s:.2f})")
        col = {"CHALLAN": RED, "REVIEW": ORANGE, "EXEMPT": BLUE}[it["decision"]]
        it["evidence"] = evidence(f"speed_{it['_speed_track']}", it["t"], it["box"], col, it["decision"].lower())
        it.pop("_speed_track")

items.sort(key=lambda it: ({"CHALLAN": 0, "REVIEW": 1, "EXEMPT": 2}[it["decision"]], it["t"]))
for i, it in enumerate(items, 1):
    it["id"] = f"{vp.stem}-{i:03d}"
summary = {"video": vp.name, "checks": checks, "challans": sum(it["decision"] == "CHALLAN" for it in items),
           "review": sum(it["decision"] == "REVIEW" for it in items), "exempt": sum(it["decision"] == "EXEMPT" for it in items),
           "notes": notes}
(out / "challans.json").write_text(json.dumps({"summary": summary, "items": items}, indent=1))
md = [f"# e-challan report: {vp.name}", "",
      f"Challans: **{summary['challans']}** · for human review: **{summary['review']}** · emergency vehicles exempted: **{summary['exempt']}**", ""]
for it in items:
    md += [f"## {it['id']} — {it['decision']}: {it['violation']} ({it['vehicle']}, {it['t']:.1f} s, confidence {it['confidence']:.2f})"]
    md += [f"- {r}" for r in it["reasons"]]
    if it.get("evidence"):
        md.append(f"- evidence: `{it['evidence']}`")
    md.append("")
if notes:
    md += ["## Not checked / notes"] + [f"- {n}" for n in notes]
(out / "challans.md").write_text("\n".join(md))
print(json.dumps(summary))
