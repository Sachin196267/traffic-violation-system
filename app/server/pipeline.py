"""
Analysis pipeline: original video -> selected detectors (multi-frame, tracked) -> findings -> evidence ->
confirmed (auto challan) / review (human queue) / exempt (emergency vehicle). Runs in a background worker.
"""
import hashlib
import json
import shutil
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

import config
import db
from ml import plate
from ml.detectors import LANES, REGISTRY

POOL = ThreadPoolExecutor(max_workers=1)    # one analysis at a time: the models share the Mac's GPU
LANE_POOL = ThreadPoolExecutor(max_workers=1)   # lane estimation (context only) runs after the results are ready
_lock = threading.Lock()


def set_status(aid, **kw):
    with db.conn() as c:
        cols = ", ".join(f"{k}=?" for k in kw)
        c.execute(f"UPDATE analyses SET {cols} WHERE id=?", (*kw.values(), aid))


def h264(src: Path, dst: Path, start=None, dur=None, width=1280, bitrate="3M"):
    """browser-playable copy (the tools write mp4v)"""
    cmd = ["ffmpeg", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{max(0, start):.2f}"]
    cmd += ["-i", str(src)]
    if dur is not None:
        cmd += ["-t", f"{dur:.2f}"]
    # playback copies: at most 1280 px wide, fastest preset (a 4K original is kept untouched for evidence)
    cmd += ["-vf", f"scale='min({width},iw)':-2", "-c:v", "h264_videotoolbox", "-b:v", bitrate, "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(dst)]
    ok = subprocess.run(cmd).returncode == 0 and dst.exists()
    if not ok:   # fall back to software encoding
        cmd = [c if c != "h264_videotoolbox" else "libx264" for c in cmd]
        i = cmd.index("-b:v"); cmd[i:i + 2] = ["-preset", "ultrafast", "-crf", "23"]
        ok = subprocess.run(cmd).returncode == 0 and dst.exists()
    return ok


def probe(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fps": round(fps, 3), "frames": n, "duration": round(n / fps, 2),
            "width": int(cap.get(3)), "height": int(cap.get(4))}


COL = {"confirmed": (40, 40, 220), "review": (0, 160, 245), "exempt": (220, 150, 30)}


def evidence(video: Path, f, fid: str, ev: Path, annotated: Path | None):
    """frames at t-0.6 / t / t+0.6 (raw + boxed), clip of the original, clip of the annotated tool video"""
    ev.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    frames = []
    for k, dt in enumerate((-0.6, 0.0, 0.6)):
        t = max(0.0, f.t + dt)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        raw = ev / f"frame{k}_raw.jpg"; cv2.imwrite(str(raw), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        item = {"t": round(t, 2), "raw": raw.name}
        if f.box and dt == 0.0:
            x1, y1, x2, y2 = map(int, f.box)
            b = fr.copy()
            cv2.rectangle(b, (x1, y1), (x2, y2), COL[f.route], 3)
            cv2.putText(b, f"{f.track}  {f.violation}  {f.confidence:.2f}", (x1, max(22, y1 - 8)), 0, 0.7, COL[f.route], 2)
            p = ev / f"frame{k}_box.jpg"; cv2.imwrite(str(p), b, [cv2.IMWRITE_JPEG_QUALITY, 92]); item["boxed"] = p.name
            H, W = fr.shape[:2]; m = int(max(x2 - x1, y2 - y1) * (3 if f.detector == "helmet" else 1.4)) + 40
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 + (m // 5 if f.detector == "helmet" else 0)
            crop = fr[max(0, cy - m // 2):min(H, cy + m // 2), max(0, cx - m // 2):min(W, cx + m // 2)]
            if crop.size:
                p = ev / "crop.jpg"; cv2.imwrite(str(p), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]); item["crop"] = p.name
        frames.append(item)
    out = {"frames": frames}
    if h264(video, ev / "clip.mp4", f.t_start, max(2.0, f.t_end - f.t_start)):
        out["clip"] = "clip.mp4"
    if annotated and annotated.exists() and h264(annotated, ev / "clip_annotated.mp4", f.t_start, max(2.0, f.t_end - f.t_start)):
        out["clip_annotated"] = "clip_annotated.mp4"
    out["sha256"] = fingerprint(ev)
    return out


def fingerprint(ev: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(ev.iterdir()):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def challan_number():
    return "TVP-" + time.strftime("%Y%m%d") + "-" + db.new_id()[:6].upper()


def create_challan(finding_id, analysis_id, source, reviewer_id=None, reviewer_plate=None, offences=None):
    """exactly one challan per vehicle: never a second challan for the same case (unique index too). Charged:
    the case's offence + other certain offences of the same vehicle (automatic), or what the reviewer ticked."""
    with db.conn() as c:
        old = c.execute("SELECT id FROM challans WHERE finding_id=?", (finding_id,)).fetchone()
        if old:
            return old["id"]
        f = db.row(c.execute("SELECT * FROM findings WHERE id=?", (finding_id,)).fetchone(), db.FINDING_JSON)
        allv = [f["violation"]] + [o["violation"] for o in f["offences"] or []]
        if offences is None:
            offences = [f["violation"]] + [o["violation"] for o in f["offences"] or [] if o.get("certain")]
        offences = [v for v in allv if v in offences] or [f["violation"]]
        cid = db.new_id()
        c.execute("INSERT INTO challans (id,number,finding_id,analysis_id,created,source,reviewer_id,status,plate,evidence_hash,offences) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (cid, challan_number(), finding_id, analysis_id, db.now(), source, reviewer_id, "issued",
                   ((f["evidence"] or {}).get("plate") or {}).get("plate") or reviewer_plate,   # READ plate, else reviewer's
                   (f["evidence"] or {}).get("sha256", ""), json.dumps(offences)))
        c.execute("UPDATE findings SET status='challan' WHERE id=?", (finding_id,))
    return cid


def run_analysis(aid: str):
    with db.conn() as c:
        a = db.row(c.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone(), ("options", "meta"))
    if not a or a["status"] == "cancelled":
        return                                   # deleted before it started
    work = config.ANALYSES / aid
    video = work / "original" / a["filename"]
    opts = a["options"]
    try:
        meta = probe(video)
        set_status(aid, status="processing", stage="Preparing video", progress=0.02, meta=json.dumps(meta))
        orig = video
        # (camera setups / speed calibrations are in original pixels: keep the original for those)
        if (max(meta["width"], meta["height"]) > 1920 or meta["fps"] > 31) and not (opts.get("camera") or opts.get("speed_calib")):
            set_status(aid, stage="Preparing a 1080p working copy", progress=0.03)
            proc = work / "proc.mp4"
            cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video)] + (["-t", str(opts["max_sec"])] if opts.get("max_sec") else []) + \
                  ["-vf", "scale=1920:-2,fps=30" if meta["fps"] > 31 else "scale=1920:-2", "-c:v", "h264_videotoolbox", "-b:v", "12M", "-an", str(proc)]
            if subprocess.run(cmd).returncode == 0 and proc.exists():
                video = proc
                meta.update(probe(video), original=probe(orig), working_copy="1080p (models run on this; original kept)")
                set_status(aid, meta=json.dumps(meta))
        # playable copy in the background: detection does not wait for it
        pl = threading.Thread(target=h264, args=(video, work / "playable.mp4", None, opts.get("max_sec")), daemon=True); pl.start()
        steps = [k for k in opts["checks"] if k in REGISTRY] + (["lanes"] if opts.get("lanes", True) else [])
        # shared vehicle detection + tracking, run once; helmet / triple / plates replay it (identical results)
        # every tool process: models on the Mac GPU (Ultralytics would otherwise use the CPU) - identical results
        import os
        import torch
        base_env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent / "ml/hook")}
        if torch.backends.mps.is_available():
            base_env["TVP_DEVICE"] = "mps"
        opts["_env"] = base_env
        users = [k for k in steps if k in ("helmet", "triple")] + (["plates"] if plate.READY and opts.get("plates", True) else [])
        if len(users) >= 2:
            set_status(aid, stage="Detecting and tracking vehicles (shared by all checks)", progress=0.05)
            cache = work / "det_cache.pkl"
            cmd = [config.PY, str(Path(__file__).parent / "ml/shared_detect.py"), str(video), str(cache)] + \
                  (["--max-sec", str(opts["max_sec"])] if opts.get("max_sec") else [])
            with open(work / "shared.log", "w") as fh:
                if subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=base_env).returncode == 0 and cache.exists():
                    opts["_shared_env"] = {**base_env, "TVP_DET_CACHE": str(cache)}
        # Scheduling (same models, same inputs -> identical results; the GPU is the bottleneck, so nothing else may
        # wait for it): checks run side by side; as soon as one finishes, its evidence clips and playback copy are
        # made (media engine / CPU) and plates are read for ITS flagged vehicles while the other checks still run.
        # Lanes are context only and run after the results are ready.
        artefacts, notes, n_find = {}, [], 0
        env = opts.get("_shared_env") or opts.get("_env")
        side = ThreadPoolExecutor(max_workers=4)          # evidence frames / clips, playback copies
        side.submit(motion_print, video, work, opts.get("max_sec"))   # recognises re-uploads of this footage
        plate_pool = ThreadPoolExecutor(max_workers=1)    # plate reader batches, one at a time
        frames = load_cache(work / "det_cache.pkl")
        read_ids, plate_futs, items = set(), [], []
        checks = [k for k in steps if k != "lanes"]

        def run_one(key):
            det = REGISTRY[key]
            ok, why = det.available()
            if not ok:
                return key, None, f"{det.title}: skipped ({why})"
            return key, det.run(video, work, opts, None), None

        set_status(aid, stage="Running " + ", ".join(REGISTRY[k].title for k in checks), progress=0.08)
        done_n = 0
        with ThreadPoolExecutor(max_workers=len(checks) or 1) as ex:
            for fut in as_completed([ex.submit(run_one, k) for k in checks]):
                key, res, note = fut.result()
                done_n += 1
                set_status(aid, stage=f"{done_n} of {len(checks)} checks finished", progress=0.08 + 0.74 * done_n / len(checks))
                if note:
                    notes.append(note)
                if res is None:
                    continue
                findings, arts = res
                notes += arts.pop("_notes", [])
                ann = arts.get("annotated")
                ann = Path(ann) if ann and Path(ann).exists() else None
                if ann:
                    side.submit(h264, ann, work / f"{key}_annotated.mp4")
                    artefacts[key] = f"{key}_annotated.mp4"
                ids = set()
                for f in findings:
                    fid = db.new_id()
                    tid = cache_track(f, frames, meta["fps"])
                    items.append((f, fid, side.submit(evidence, video, f, fid, work / "evidence" / fid, ann), tid))
                    if tid is not None:
                        ids.add(tid)
                new = ids - read_ids
                if plate.READY and opts.get("plates", True) and new:
                    # flagged vehicles + vehicles overlapping them (the reader's "same plate on two vehicles" guard)
                    batch = (new | neighbours(new, frames)) - read_ids
                    read_ids |= batch
                    plate_futs.append(plate_pool.submit(plate.run, video, work, opts.get("max_sec"), env, tracks=sorted(batch), tag=key))
        plates_v = []
        if plate_futs:
            set_status(aid, stage="Finishing number-plate reading", progress=0.84)
            for pf in plate_futs:
                plates_v += pf.result()
            plates_v = plate.dedupe(plates_v)             # the reader's duplicate guard, across all batches
        set_status(aid, stage="Collecting evidence", progress=0.9)
        for f, fid, evf, tid in items:
            ev = evf.result()
            pl_ = plate.link(plates_v, f.t, f.box, head=f.detector == "helmet", track=tid)
            if pl_.get("crop") and Path(pl_["crop"]).exists():
                shutil.copy(pl_["crop"], work / "evidence" / fid / "plate.jpg"); pl_["crop"] = "plate.jpg"
            ev["plate"] = pl_
            ev["vehicle_track"] = tid
            ev["sha256"] = fingerprint(work / "evidence" / fid)
            status = {"confirmed": "confirmed", "review": "pending_review", "exempt": "exempt"}[f.route]
            with db.conn() as c:
                c.execute("INSERT INTO findings (id,analysis_id,detector,violation,track,t,t_start,t_end,confidence,model_decision,route,"
                          "status,box,reasons,uncertain_reasons,evidence,emergency,vehicle,model_info,created) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (fid, aid, f.detector, f.violation, f.track, f.t, f.t_start, f.t_end, f.confidence,
                           f.model_decision, f.route, status, json.dumps(f.box), json.dumps(f.reasons),
                           json.dumps(f.uncertain_reasons), json.dumps(ev), f.emergency, f.vehicle,
                           json.dumps(f.model_info), db.now()))
            n_find += 1
        side.shutdown(wait=True)
        pl.join()
        set_status(aid, stage="Merging duplicate detections", progress=0.95)
        n_dup, n_seen = merge_duplicates(aid, work)
        with db.conn() as c:
            conf_ids = [r["id"] for r in c.execute("SELECT id FROM findings WHERE analysis_id=? AND status='confirmed'", (aid,))]
        for fid in conf_ids:
            create_challan(fid, aid, "automatic")
        if n_dup:
            notes.append(f"{n_dup} repeat detection(s) of the same vehicle merged - one challan or review per vehicle")
        if n_seen:
            notes.append(f"{n_seen} detection(s) were already handled in an earlier upload of this footage - not challaned again")
        meta.update(artefacts=artefacts, notes=notes, plate={"ready": plate.READY, "status": plate.STATUS,
                                                              "vehicles": len(plates_v),
                                                              "read": sum(v["decision"] == "READ" for v in plates_v)})
        if "lanes" in steps:
            meta["lanes_pending"] = True
        set_status(aid, status="done", stage="Done", progress=1.0, meta=json.dumps(meta), finished=db.now())
        if "lanes" in steps:
            LANE_POOL.submit(run_lanes, aid, video, work, opts)
    except Exception as e:
        with db.conn() as c:
            r = c.execute("SELECT status FROM analyses WHERE id=?", (aid,)).fetchone()
        if not r or r["status"] == "cancelled":
            return                               # deleted by the user while running
        try:
            (work / "error.log").write_text(traceback.format_exc())
        except OSError:
            pass
        set_status(aid, status="failed", stage="Failed", error=str(e)[:500], finished=db.now())


ROUTE_RANK = {"confirmed": 0, "review": 1, "exempt": 2}


def load_cache(p: Path):
    if not p.exists():
        return []
    import pickle
    with open(p, "rb") as fh:
        return pickle.load(fh)["frames"]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def cache_track(f, frames, fps):
    """track id (shared detection pass) of the vehicle behind a finding"""
    import re
    m = re.match(r"[HT](\d+)", f.track or "")
    if f.detector in ("helmet", "triple") and m:
        return int(m.group(1))                           # these tools replay the shared pass: same ids
    if not f.box or not frames:
        return None
    fr = min(frames, key=lambda d: abs(d["fi"] / fps - f.t))
    if fr["id"] is None:
        return None
    best = max(((float(_iou(f.box, b)), int(i)) for b, i in zip(fr["xyxy"].tolist(), fr["id"].tolist())), default=(0, None))
    return best[1] if best[0] >= 0.3 else None


def neighbours(ids, frames):
    """vehicles whose boxes overlap a flagged vehicle in any frame"""
    out = set()
    for d in frames:
        if d["id"] is None:
            continue
        B = dict(zip(d["id"].astype(int).tolist(), d["xyxy"].tolist()))
        for v in ids & B.keys():
            for u, b in B.items():
                if u != v and _iou(B[v], b) > 0.02:
                    out.add(u)
    return out


def motion_print(video, work, max_sec=None):
    """tiny grey 64x36 frames at 5 per second: recognises the same footage in another upload (align)"""
    p = work / "motion.npy"
    if p.exists():
        return np.load(p)
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox"] + (["-t", str(max_sec)] if max_sec else []) + \
          ["-i", str(video), "-vf", "fps=5,scale=64:36,format=gray", "-f", "rawvideo", "-"]
    g = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.uint8).reshape(-1, 36 * 64)
    np.save(p, g)
    return g


def align(ga, gb, min_overlap=25):
    """same footage? -> (k, score): sample i of A is sample i+k of B (0.2 s steps), else None.
    Compares motion (frame-to-frame change), not the picture itself: the same fixed camera on another day
    shows the same road, but never the same moving traffic."""
    def sig(g):
        d = np.diff(g.astype(np.float32), axis=0)
        d -= d.mean(1, keepdims=True)
        n = np.linalg.norm(d, axis=1)
        return d / np.maximum(n, 1e-6)[:, None], (n > 40).astype(np.float32)   # rows with real motion
    if len(ga) < min_overlap + 1 or len(gb) < min_overlap + 1:
        return None
    (A, ma), (B, mb) = sig(ga), sig(gb)
    A, B = A * ma[:, None], B * mb[:, None]
    n, m = len(A), len(B)
    acc, cnt = np.zeros(n + m - 1), np.zeros(n + m - 1)
    for s in range(0, n, 2000):
        C = A[s:s + 2000] @ B.T
        for r in range(len(C)):
            if ma[s + r]:
                o = n - 1 - (s + r)
                acc[o:o + m] += C[r]; cnt[o:o + m] += mb
    ok = cnt >= min_overlap
    if not ok.any():
        return None
    score = np.where(ok, acc / np.maximum(cnt, 1), -1)
    o = int(np.argmax(score))
    return (o - (n - 1), float(score[o])) if score[o] >= 0.5 else None


def vehicle_chains(frames, fps):
    """track id -> chain root: a track that starts <= 1 s after another ended, in (almost) the same place and with
    a similar size, is the same vehicle re-found by the tracker"""
    first, last = {}, {}
    for d in frames:
        if d["id"] is None:
            continue
        t = d["fi"] / fps
        for b, i in zip(d["xyxy"].tolist(), d["id"].astype(int).tolist()):
            first.setdefault(i, (t, b)); last[i] = (t, b)
    parent = {i: i for i in first}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    ends = sorted(last.items(), key=lambda kv: kv[1][0])
    starts = sorted(first.items(), key=lambda kv: kv[1][0])
    j0 = 0
    for i, (te, be) in ends:
        while j0 < len(starts) and starts[j0][1][0] < te:
            j0 += 1
        for k in range(j0, len(starts)):
            jd, (ts, bs) = starts[k]
            if ts - te > 1.0:
                break
            if jd == i:
                continue
            ae, as_ = (be[2] - be[0]) * (be[3] - be[1]), (bs[2] - bs[0]) * (bs[3] - bs[1])
            if _iou(be, bs) > 0.3 and 0.5 < as_ / max(ae, 1) < 2:
                parent[root(jd)] = root(i)
    return {i: root(i) for i in parent}


def _vehicle(f):
    """shared-pass track id of a finding's vehicle (stored at analysis time; older analyses: from the track label)"""
    import re
    v = (f.get("evidence") or {}).get("vehicle_track")
    if v is not None:
        return int(v)
    m = re.match(r"[HT](\d+)", f.get("track") or "")
    return int(m.group(1)) if m and f["detector"] in ("helmet", "triple") else None


OPEN = ("challan", "confirmed", "pending_review", "needs_evidence")   # an issued challan or a case still open
DUP = "duplicate: "                                                    # prefix of the merge reason (removed on restore)


def _footage(aid):
    """one analysis, as needed to compare its vehicles with another upload of the same footage"""
    work = config.ANALYSES / aid
    with db.conn() as c:
        a = c.execute("SELECT filename, created, meta FROM analyses WHERE id=?", (aid,)).fetchone()
        F = [db.row(r, db.FINDING_JSON) for r in c.execute("SELECT * FROM findings WHERE analysis_id=? AND status<>'duplicate'", (aid,))]
    meta = json.loads((a["meta"] if a else None) or "{}")
    fps, W, H = meta.get("fps") or 25, meta.get("width") or 1, meta.get("height") or 1
    frames = load_cache(work / "det_cache.pkl")
    chain = vehicle_chains(frames, fps) if frames else {}
    paths = {}                                   # vehicle -> {0.2 s sample: box in 0..1 picture coordinates}
    for d in frames:
        if d["id"] is not None:
            k = round(d["fi"] / fps * 5)
            for bx, i in zip(d["xyxy"].tolist(), d["id"].astype(int).tolist()):
                paths.setdefault(chain.get(i, i), {})[k] = (bx[0] / W, bx[1] / H, bx[2] / W, bx[3] / H)
    veh = {f["id"]: (chain.get(v, v) if v is not None else None) for f in F for v in [_vehicle(f)]}
    return dict(aid=aid, work=work, F=F, W=W, H=H, paths=paths, veh=veh,
                filename=a["filename"] if a else "?", created=a["created"] if a else 0)


def _same_across(f, X, g, Y, k):
    """same vehicle in two uploads of one footage (sample i of X = sample i+k of Y)?"""
    px, py = X["paths"].get(X["veh"].get(f["id"])), Y["paths"].get(Y["veh"].get(g["id"]))
    if px and py:                                # both tracked: the whole path must coincide
        ious = [_iou(b, py[i + k]) for i, b in px.items() if i + k in py]
        return len(ious) >= 3 and float(np.median(ious)) >= 0.5
    if f["detector"] != g["detector"] or not (f.get("box") and g.get("box")):
        return False                             # older analyses: same check, same moment, same place
    if abs(f["t"] + k * 0.2 - g["t"]) > 0.6:
        return False
    nx = [v / (X["W"], X["H"])[j % 2] for j, v in enumerate(f["box"])]
    ny = [v / (Y["W"], Y["H"])[j % 2] for j, v in enumerate(g["box"])]
    return _iou(nx, ny) >= 0.4


def merge_duplicates(aid, work):
    """ONE challan or review per vehicle. Detections are one vehicle when
      (a) same vehicle in this video: same shared track, or the tracker re-finding it (vehicle_chains) - e.g. rider
          and pillion without helmet, the same rider counted twice, or triple riding + no helmet on one motorcycle,
      (b) same number plate read with certainty,
      (c) same violation <= 8 s apart with near-identical close-ups (CLIP similarity >= 0.92), or
      (d) another upload of the SAME footage (align) shows the same vehicle at the same moment.
    One case is kept per vehicle: a reviewer's challan > an automatic challan > a case waiting for more evidence >
    the most confident case waiting for review (earlier upload first). Every other open detection of that vehicle
    becomes 'duplicate' (kept for audit, linked, never challaned; a second challan is removed); a different
    offence is listed on the kept case (certain ones are charged on its challan). Decided cases stay decided:
    a re-detection of an offence a reviewer rejected, or of an exempt emergency vehicle, is merged into it.
    Returns (repeats merged in this video, events already handled in another upload)."""
    X = _footage(aid)
    groups = [X]
    src = work / "playable.mp4"
    mine = motion_print(src, work) if (work / "motion.npy").exists() or src.exists() else None
    offs = {}
    if mine is not None:
        with db.conn() as c:
            others = [r["id"] for r in c.execute("SELECT id FROM analyses WHERE status='done' AND id<>? ORDER BY created", (aid,))]
        for o in others:
            ow = config.ANALYSES / o
            src = ow / "playable.mp4"
            if not (ow / "motion.npy").exists() and not src.exists():
                continue
            r = align(mine, motion_print(src, ow))
            if r:
                offs[o] = r[0]; groups.append(_footage(o))
    nodes = [(G, f) for G in groups for f in G["F"] if f["status"] in OPEN + ("rejected", "exempt")]
    if len(nodes) < 2:
        return 0, 0
    with db.conn() as c:
        ch = {r["finding_id"]: dict(r) for r in c.execute(
            "SELECT * FROM challans WHERE analysis_id IN (%s)" % ",".join("?" * len(groups)), [G["aid"] for G in groups])}
    parent = list(range(len(nodes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def join(i, j):
        parent[root(j)] = root(i)
    plate_of = lambda f: ((f["evidence"] or {}).get("plate") or {}).get("plate")
    crops = {f["id"]: X["work"] / "evidence" / f["id"] / "crop.jpg" for f in X["F"]}
    have = [i for i, pth in crops.items() if pth.exists()]
    emb = {}
    if len(have) >= 2:
        import sys
        sys.path.insert(0, str(config.ROOT / "ambulance_training"))
        from clip_features import Clip
        emb = dict(zip(have, Clip().embed([str(crops[i]) for i in have])))
    for i, (G, f) in enumerate(nodes):
        for j in range(i):
            H_, g = nodes[j]
            if G is H_:                          # same video
                if G["veh"].get(f["id"]) is not None and G["veh"].get(f["id"]) == G["veh"].get(g["id"]):
                    join(j, i)
                elif plate_of(f) and plate_of(f) == plate_of(g):
                    join(j, i)
                elif f["violation"] == g["violation"] and abs(f["t"] - g["t"]) <= 8 and f["id"] in emb and g["id"] in emb \
                        and float(emb[f["id"]] @ emb[g["id"]]) >= 0.92:
                    join(j, i)
            elif G is X or H_ is X:              # this upload vs another upload of the same footage
                (A, fa), (B, fb) = ((G, f), (H_, g)) if G is X else ((H_, g), (G, f))
                if _same_across(fa, A, fb, B, offs[B["aid"]]):
                    join(j, i)
    comps = {}
    for i in range(len(nodes)):
        comps.setdefault(root(i), []).append(nodes[i])

    def rank(Gf):
        G, f = Gf
        c_ = ch.get(f["id"])
        r = 0 if (c_ and c_["source"] == "human_review") else 1 if (c_ or f["status"] in ("challan", "confirmed")) \
            else 2 if f["status"] == "needs_evidence" else 3
        return (r, G["created"], -(f["confidence"] or 0))
    is_human = lambda x: (ch.get(x[1]["id"]) or {}).get("source") == "human_review"
    n_same = n_cross = 0
    for comp in comps.values():
        if len(comp) < 2:
            continue
        open_ = sorted([x for x in comp if x[1]["status"] in OPEN], key=rank)
        exempt = [x for x in comp if x[1]["status"] == "exempt"]
        rej = {}
        for x in comp:
            if x[1]["status"] == "rejected":
                rej.setdefault(x[1]["violation"], x)
        moves = []                               # (detection, the case it joins, why)
        if exempt and not any(map(is_human, open_)):
            moves = [(x, exempt[0], "emergency vehicle - never challaned") for x in open_ + exempt[1:]]
        else:
            live = []
            for x in open_:                      # a reviewer's "no violation" wins over any re-detection
                if x[1]["violation"] in rej and not is_human(x):
                    moves.append((x, rej[x[1]["violation"]], "a reviewer already rejected this offence for this vehicle"))
                else:
                    live.append(x)
            moves += [(x, live[0], None) for x in live[1:]]
        for (G, f), (TG, T), why in moves:
            other = "" if TG is G else f" in another upload of this footage ({TG['filename']})"
            if why is None:
                why = (f"same vehicle{other}: offence '{f['violation']}' is included in that case"
                       if f["violation"] != T["violation"] else f"same vehicle{other}" if other
                       else "same vehicle (another person on it, or the same one seen again)")
            K, KG = T, TG
            with db.conn() as c:
                c.execute("UPDATE findings SET status='duplicate', duplicate_of=?, uncertain_reasons=? WHERE id=?",
                          (T["id"], json.dumps((f["uncertain_reasons"] or []) + [DUP + why]), f["id"]))
                c.execute("UPDATE findings SET duplicate_of=? WHERE duplicate_of=?", (T["id"], f["id"]))   # re-point
                certain = f["id"] in ch or f["status"] in ("challan", "confirmed")
                if f["id"] in ch:                # a second challan for the same vehicle
                    c.execute("DELETE FROM challans WHERE finding_id=?", (f["id"],))
                if T["status"] in OPEN:          # the vehicle's one open case collects the offence
                    offences = list(K["offences"] or [])
                    new = [dict(id=f["id"], violation=f["violation"], confidence=f["confidence"], certain=certain,
                                t=f["t"], analysis_id=G["aid"])] if f["violation"] != K["violation"] else []
                    new += [o for o in (f["offences"] or [])]                  # offences it had collected
                    for o in new:
                        if o["violation"] != K["violation"] and all(o["violation"] != q["violation"] for q in offences):
                            offences.append(o)
                        elif o.get("certain"):
                            for q in offences:
                                if q["violation"] == o["violation"]:
                                    q["certain"] = True
                    reasons = list(K["reasons"] or [])
                    slot = lambda x: (x["track"] or "").partition(".")                 # H<track>.<rider slot>
                    if f["violation"] == K["violation"] and f["detector"] == "helmet" and TG is G and slot(f)[2] != slot(K)[2]:
                        note = "another person on this vehicle was also detected without a helmet"
                        if note not in reasons:
                            reasons.append(note)
                    K["offences"], K["reasons"] = offences, reasons
                    upd = {"offences": json.dumps(offences), "reasons": json.dumps(reasons)}
                    if not plate_of(K) and plate_of(f) and TG is G:      # plate read certainly on the other detection
                        ev = dict(K["evidence"] or {}); ev["plate"] = (f["evidence"] or {}).get("plate")
                        src = G["work"] / "evidence" / f["id"] / "plate.jpg"
                        if src.exists():
                            shutil.copy(src, KG["work"] / "evidence" / K["id"] / "plate.jpg")
                        K["evidence"] = ev; upd["evidence"] = json.dumps(ev)
                    c.execute("UPDATE findings SET " + ",".join(f"{k}=?" for k in upd) + " WHERE id=?", (*upd.values(), K["id"]))
                    kc = c.execute("SELECT * FROM challans WHERE finding_id=?", (K["id"],)).fetchone()
                    if kc:                               # certain offences are charged on the vehicle's one challan
                        charged = json.loads(kc["offences"] or "null") or [K["violation"]]
                        charged += [o["violation"] for o in offences if o.get("certain") and o["violation"] not in charged]
                        c.execute("UPDATE challans SET offences=?, plate=COALESCE(plate, ?) WHERE id=?",
                                  (json.dumps(charged), plate_of(K), kc["id"]))
            if TG is G:
                n_same += 1
            else:
                n_cross += 1
    return n_same, n_cross


def restore_after_delete(deleted_ids):
    """an analysis was deleted: detections elsewhere that were merged into its cases become cases again"""
    if not deleted_ids:
        return
    q = ",".join("?" * len(deleted_ids))
    with db.conn() as c:
        rows = [db.row(r, db.FINDING_JSON) for r in c.execute(f"SELECT * FROM findings WHERE duplicate_of IN ({q})", list(deleted_ids))]
        for f in rows:
            c.execute("UPDATE findings SET status=?, duplicate_of=NULL, uncertain_reasons=? WHERE id=?",
                      ({"confirmed": "confirmed", "review": "pending_review", "exempt": "exempt"}[f["route"]],
                       json.dumps([r for r in (f["uncertain_reasons"] or []) if not r.startswith(DUP)]), f["id"]))
        for r in c.execute("SELECT id, offences FROM findings WHERE offences IS NOT NULL").fetchall():
            o = json.loads(r["offences"] or "[]")
            if any(x["id"] in deleted_ids for x in o):
                c.execute("UPDATE findings SET offences=? WHERE id=?", (json.dumps([x for x in o if x["id"] not in deleted_ids]), r["id"]))
    for a in sorted({f["analysis_id"] for f in rows}):
        merge_duplicates(a, config.ANALYSES / a)
        with db.conn() as c:
            ids = [r["id"] for r in c.execute("SELECT id FROM findings WHERE analysis_id=? AND status='confirmed'", (a,))]
        for fid in ids:
            create_challan(fid, a, "automatic")


def run_lanes(aid, video, work, opts):
    """lane estimation after the results are shown (context only; never creates a finding)"""
    art = {}
    try:
        ctx = LANES.run(video, work, opts, None)
        art = {"lanes": {k: str(Path(v).relative_to(work)) for k, v in ctx.files.items()}, "lanes_summary": ctx.summary}
        if "video" in ctx.files:
            h264(Path(ctx.files["video"]), work / "lanes_playable.mp4")
    except Exception:
        art = {"lanes_summary": {"ok": False, "note": "lane estimation failed"}}
    with db.conn() as c:
        r = c.execute("SELECT meta FROM analyses WHERE id=?", (aid,)).fetchone()
        if not r:
            return
        m = json.loads(r["meta"] or "{}")
        m.setdefault("artefacts", {}).update(art)
        m["lanes_pending"] = False
        c.execute("UPDATE analyses SET meta=? WHERE id=?", (json.dumps(m), aid))


def kill_leftovers(aid):
    """tool processes still running for this analysis from before a server restart"""
    subprocess.run(["pkill", "-f", f"analyses/{aid}/"])


def submit(aid):
    POOL.submit(run_analysis, aid)
