"""
Signal-timing simulation for an analysed junction video. Every number it produces is SIMULATED.

measure(): traffic of the video from the shared detection + tracking pass (det_cache.pkl), fragments of one vehicle
           merged with pipeline.vehicle_chains, split by direction of travel in the image (a camera sees both
           directions; only one of them approaches the signal).
start():   measured approach (+ the user's assumptions for the other three) -> the Unity simulator (signal_sim/,
           headless, on a private copy so an open editor does not block it) -> every controller on identical arrivals.
"""
import glob
import json
import os
import pickle
import shutil
import subprocess
import threading
import time
from collections import Counter

import config

SIM_SRC = config.ROOT / "signal_sim"
RUNNER = config.DATA / "signal_sim_runner"
TYPES = ["Bike", "Bicycle", "Car", "Auto", "Bus", "Truck"]          # signal_sim Specs.Mix order
CLS = {"motorcycle": "Bike", "bicycle": "Bicycle", "car": "Car", "caravan": "Car", "vehicle fallback": "Car",
       "autorickshaw": "Auto", "bus": "Bus", "truck": "Truck", "trailer": "Truck"}
GROUPS = {"toward": "moving down the image (towards the camera)", "away": "moving up the image (away from the camera)",
          "all": "all moving vehicles"}
_lock = threading.Lock()
_active = set()   # analyses with a simulation thread alive in this server process


def unity_path():
    p = os.environ.get("TVP_UNITY")
    if p and os.path.exists(p):
        return p
    found = sorted(glob.glob("/Applications/Unity/Hub/Editor/*/Unity.app/Contents/MacOS/Unity"))
    return found[-1] if found else None


def ready():
    return SIM_SRC.exists() and unity_path() is not None


def _dir(aid):
    return config.ANALYSES / aid / "signal_sim"


def measure(aid, fps):
    cache = config.ANALYSES / aid / "det_cache.pkl"
    if not cache.exists():
        return None
    import pipeline
    with open(cache, "rb") as fh:
        c = pickle.load(fh)
    names, frames = c["names"], [d for d in c["frames"] if d["id"] is not None]
    if not frames:
        return None
    chain = pipeline.vehicle_chains(frames, fps)
    first, last, types, height = {}, {}, {}, 1.0
    for d in frames:
        for b, k, i in zip(d["xyxy"].tolist(), d["cls"].astype(int).tolist(), d["id"].astype(int).tolist()):
            t = CLS.get(names[k])
            if t is None:
                continue
            v = chain.get(i, i)
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            height = max(height, b[3])
            first.setdefault(v, (d["fi"], cx, cy)); last[v] = (d["fi"], cx, cy)
            types.setdefault(v, Counter())[t] += 1
    seconds = (frames[-1]["fi"] - frames[0]["fi"]) / fps + c.get("step", 1) / fps
    groups = {g: Counter() for g in GROUPS}
    parked = 0
    for v in first:
        dy = last[v][2] - first[v][2]
        if last[v][0] == first[v][0] or abs(dy) < 0.03 * height and abs(last[v][1] - first[v][1]) < 0.03 * height:
            parked += 1   # seen once or not moving: not part of the flow
            continue
        t = types[v].most_common(1)[0][0]
        groups["toward" if dy > 0 else "away"][t] += 1
        groups["all"][t] += 1
    out = {"seconds": round(seconds, 1), "parked_or_single_frame": parked, "groups": {}}
    for g, cnt in groups.items():
        n = sum(cnt.values())
        out["groups"][g] = {"label": GROUPS[g], "vehicles": n, "per_hour": round(n / seconds * 3600) if seconds > 0 else 0,
                            "mix": {t: round(cnt[t] / n, 3) if n else 0 for t in TYPES}, "counts": {t: cnt[t] for t in TYPES}}
    return out


def state(aid):
    d = _dir(aid)
    st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {"status": "none"}
    if st.get("status") in ("queued", "running") and aid not in _active:   # the server restarted mid-run
        st.update(status="failed", error="interrupted (the server restarted) - run it again")
        _status(aid, status="failed", error=st["error"])
    if (d / "result.json").exists() and st.get("status") == "done":
        st["result"] = json.loads((d / "result.json").read_text())
    if (d / "job.json").exists():
        st["job"] = json.loads((d / "job.json").read_text())
    return st


def _status(aid, **kw):
    d = _dir(aid); d.mkdir(parents=True, exist_ok=True)
    cur = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
    cur.update(kw); (d / "status.json").write_text(json.dumps(cur))


def start(aid, fps, p):
    """p: group, lanes (on the filmed approach), other (3 rates E,S,W veh/h), amb_per_hour, bus_stops, cattle_per_hour, minutes"""
    m = measure(aid, fps)
    if not m:
        raise ValueError("this analysis has no vehicle tracks to measure traffic from")
    g = m["groups"].get(p.get("group", "toward"))
    if not g or g["vehicles"] < 5:
        raise ValueError("too few vehicles in the chosen direction to measure a flow (need at least 5)")
    other = [float(x) for x in p.get("other", [])]
    if len(other) != 3 or not all(0 <= x <= 2500 for x in other):
        raise ValueError("enter the other three approaches' traffic (0-2500 vehicles/hour each)")
    lanes = int(p.get("lanes", 3))
    if not 1 <= lanes <= 8:
        raise ValueError("lanes on the filmed approach must be 1-8")
    minutes = float(p.get("minutes", 20))
    if not 5 <= minutes <= 30:
        raise ValueError("simulated minutes must be 5-30")
    # the simulated junction has 2 lanes per approach: keep the per-lane flow of the filmed road
    sim_rate = round(g["per_hour"] * 2 / lanes)
    job = {"rates": [float(min(sim_rate, 2500))] + other, "mix": [g["mix"][t] for t in TYPES],
           "ambPerHour": float(min(max(p.get("amb_per_hour", 4), 0), 30)), "busStops": bool(p.get("bus_stops", True)),
           "cattlePerHour": float(min(max(p.get("cattle_per_hour", 0), 0), 20)), "minutes": minutes, "seeds": [1, 2, 3],
           "measured": {"group": p.get("group", "toward"), "vehicles": g["vehicles"], "seconds": m["seconds"], "per_hour": g["per_hour"],
                        "lanes": lanes, "sim_per_hour": min(sim_rate, 2500), "capped": sim_rate > 2500}}
    d = _dir(aid); d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").unlink(missing_ok=True)
    (d / "job.json").write_text(json.dumps(job))
    _status(aid, status="queued", started=time.time(), error=None)
    _active.add(aid)
    threading.Thread(target=_run, args=(aid,), daemon=True).start()


def _run(aid):
    d = _dir(aid)
    with _lock:   # one simulation at a time: Unity locks its project
        _status(aid, status="running")
        try:
            unity = unity_path()
            if not unity:
                raise RuntimeError("Unity editor not found (set TVP_UNITY)")
            RUNNER.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", "--delete", "--exclude", "Library", "--exclude", "Temp", "--exclude", "Logs*",
                            "--exclude", "Results", "--exclude", "UserSettings", f"{SIM_SRC}/", f"{RUNNER}/"], check=True)
            r = subprocess.run([unity, "-batchmode", "-nographics", "-projectPath", str(RUNNER), "-executeMethod", "SignalSim.SimJob.Run",
                                "-job", str(d / "job.json"), "-out", str(d / "result.json"), "-logFile", str(d / "unity.log")],
                               timeout=1800)
            if r.returncode != 0 or not (d / "result.json").exists():
                tail = (d / "unity.log").read_text(errors="replace")[-600:] if (d / "unity.log").exists() else ""
                raise RuntimeError(f"simulator exited with {r.returncode}: {tail.strip().splitlines()[-1] if tail.strip() else 'no log'}")
            _status(aid, status="done", finished=time.time())
        except Exception as e:
            _status(aid, status="failed", error=str(e)[:400])
        finally:
            _active.discard(aid)
