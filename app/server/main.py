"""
Traffic Violation Platform - API server (FastAPI). Serves the built web app from app/web/dist.

    cd app/server && ../../.venv/bin/uvicorn main:app --port 8600
"""
import json
import re
import shutil
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import challan_pdf
import config
import db
import pipeline
import security as sec
import signal_sim
from ml import plate
from ml.detectors import LANES, REGISTRY

app = FastAPI(title="Traffic Violation Platform", docs_url=None, redoc_url=None)
COOKIE = "tvp_session"


@app.on_event("startup")
def startup():
    db.init()
    with db.conn() as c:   # analyses interrupted by a restart are run again (the original video is kept)
        ids = [r["id"] for r in c.execute("SELECT id FROM analyses WHERE status IN ('queued','processing')")]
        c.execute("UPDATE analyses SET status='queued', stage='Queued (restarted)', progress=0 WHERE status='processing'")
    for i in ids:
        pipeline.kill_leftovers(i)
        pipeline.submit(i)


@app.middleware("http")
async def csrf_and_headers(request: Request, call_next):
    # state-changing API calls must come from the app itself (custom header + SameSite=Lax cookie)
    if request.url.path.startswith("/api/") and request.method in ("POST", "PUT", "DELETE") \
            and request.headers.get("x-tvp") != "1":
        return JSONResponse({"detail": "Missing request header"}, 403)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp


def user(request: Request):
    u = sec.session_user(request.cookies.get(COOKIE))
    if not u or not u["verified"]:
        raise HTTPException(401, "Please sign in")
    return u


def err(msg, code=400):
    raise HTTPException(code, msg)


def _client(request):
    return request.client.host if request.client else "?"


# ---------------- auth ----------------
class Signup(BaseModel):
    name: str
    identifier: str          # email or phone
    password: str
    captcha_token: str
    captcha: str


class Login(BaseModel):
    identifier: str
    password: str
    captcha_token: str | None = None
    captcha: str | None = None


class Code(BaseModel):
    identifier: str
    code: str
    purpose: str = "verify"


class Ident(BaseModel):
    identifier: str
    captcha_token: str | None = None
    captcha: str | None = None


class Reset(BaseModel):
    identifier: str
    code: str
    password: str


def norm_ident(s: str):
    s = s.strip().lower()
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", s):
        return "email", s
    d = re.sub(r"[\s\-()]", "", s)
    if re.fullmatch(r"\+?\d{10,13}", d):
        return "phone", d
    err("Enter a valid email address or phone number")


def find_user(ident):
    kind, v = norm_ident(ident)
    with db.conn() as c:
        return db.row(c.execute(f"SELECT * FROM users WHERE {kind}=?", (v,)).fetchone()), kind, v


_fails: dict[str, int] = {}


@app.get("/api/auth/config")
def auth_config():
    return {"dev_mode": config.DEV_MODE,
            "delivery": "developer outbox (no email/SMS provider configured)" if config.DEV_MODE else "email"}


@app.get("/api/auth/captcha")
def captcha():
    img, tok = sec.new_captcha()
    return {"image": img, "token": tok}


def _code_reply(target, purpose):
    r = sec.send_code(target, purpose)
    out = {"sent": True, "delivery": r["delivered"]}
    if config.DEV_MODE:
        out["dev_code"] = r["dev_code"]   # shown only in developer mode, clearly labelled in the UI
    return out


@app.post("/api/auth/signup")
def signup(b: Signup, request: Request):
    if sec.limited("signup:" + _client(request), 8, 3600):
        err("Too many attempts. Try again later.", 429)
    if not sec.check_captcha(b.captcha_token, b.captcha):
        err("The characters did not match. Try the new image.")
    if len(b.name.strip()) < 2:
        err("Enter your name")
    probs = sec.password_problems(b.password)
    if probs:
        err("Password needs " + ", ".join(probs))
    u, kind, v = find_user(b.identifier)
    if u and u["verified"]:
        err("An account with this " + kind + " already exists. Sign in instead.")
    with db.conn() as c:
        if u:
            c.execute("UPDATE users SET name=?, pw_hash=? WHERE id=?", (b.name.strip(), sec.hash_pw(b.password), u["id"]))
        else:
            c.execute(f"INSERT INTO users (id,name,{kind},pw_hash,verified,created) VALUES (?,?,?,?,0,?)",
                      (db.new_id(), b.name.strip(), v, sec.hash_pw(b.password), db.now()))
    return _code_reply(v, "verify")


@app.post("/api/auth/verify")
def verify(b: Code, response: Response, request: Request):
    if sec.limited("verify:" + _client(request), 20, 600):
        err("Too many attempts. Try again later.", 429)
    u, kind, v = find_user(b.identifier)
    if not u or not sec.check_code(v, "verify", b.code):
        err("That code is wrong or has expired")
    with db.conn() as c:
        c.execute("UPDATE users SET verified=1 WHERE id=?", (u["id"],))
    _set_cookie(response, request, sec.create_session(u["id"]))
    return {"ok": True}


@app.post("/api/auth/resend")
def resend(b: Ident, request: Request):
    if sec.limited("resend:" + _client(request), 5, 600):
        err("Please wait before asking for another code.", 429)
    u, kind, v = find_user(b.identifier)
    if not u:
        return {"sent": True}          # no account enumeration
    return _code_reply(v, "verify")


def _set_cookie(response, request, tok):
    response.set_cookie(COOKIE, tok, httponly=True, samesite="lax", secure=request.url.scheme == "https",
                        max_age=config.SESSION_DAYS * 86400, path="/")


@app.post("/api/auth/login")
def login(b: Login, response: Response, request: Request):
    ip = _client(request)
    if sec.limited("login:" + ip, 20, 600):
        err("Too many attempts. Try again later.", 429)
    need_captcha = _fails.get(ip, 0) >= 3
    if need_captcha and not (b.captcha_token and sec.check_captcha(b.captcha_token, b.captcha or "")):
        raise HTTPException(400, detail={"message": "Complete the check to continue", "captcha_required": True})
    u, kind, v = find_user(b.identifier)
    if not u or not sec.check_pw(b.password, u["pw_hash"]):
        _fails[ip] = _fails.get(ip, 0) + 1
        raise HTTPException(400, detail={"message": "Email/phone or password is incorrect",
                                         "captcha_required": _fails[ip] >= 3})
    if not u["verified"]:
        r = _code_reply(v, "verify")
        return {"ok": False, "needs_verification": True, **r}
    _fails.pop(ip, None)
    _set_cookie(response, request, sec.create_session(u["id"]))
    return {"ok": True}


@app.post("/api/auth/forgot")
def forgot(b: Ident, request: Request):
    if not (b.captcha_token and sec.check_captcha(b.captcha_token, b.captcha or "")):
        err("The characters did not match. Try the new image.")
    if sec.limited("forgot:" + _client(request), 5, 3600):
        err("Too many attempts. Try again later.", 429)
    u, kind, v = find_user(b.identifier)
    if not u:
        return {"sent": True}
    return _code_reply(v, "reset")


@app.post("/api/auth/reset")
def reset(b: Reset, request: Request):
    probs = sec.password_problems(b.password)
    if probs:
        err("Password needs " + ", ".join(probs))
    u, kind, v = find_user(b.identifier)
    if not u or not sec.check_code(v, "reset", b.code):
        err("That code is wrong or has expired")
    with db.conn() as c:
        c.execute("UPDATE users SET pw_hash=?, verified=1 WHERE id=?", (sec.hash_pw(b.password), u["id"]))
        c.execute("DELETE FROM sessions WHERE user_id=?", (u["id"],))
    return {"ok": True}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    sec.end_session(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(u=Depends(user)):
    return {k: u[k] for k in ("id", "name", "email", "phone", "role")}


# ---------------- system ----------------
@app.get("/api/system")
def system(u=Depends(user)):
    dets = [{"key": d.key, "title": d.title, "produces": d.produces, "needs": d.needs,
             "available": d.available()[0], "status": d.available()[1]} for d in REGISTRY.values()]
    return {"detectors": dets, "lanes": {"title": LANES.title, "role": "context only - never creates a challan"},
            "plate": {"ready": plate.READY, "status": plate.STATUS},
            "emergency": {"model": "CLIP ViT-B/16 + probe (runs/ambulance_cls)", "exempt": ["ambulance", "police", "fire"]}}


# ---------------- analyses ----------------
def _analysis(a_id, uid=None):
    with db.conn() as c:
        a = db.row(c.execute("SELECT * FROM analyses WHERE id=?", (a_id,)).fetchone(), ("options", "meta"))
    if not a:
        err("Not found", 404)
    return a


@app.post("/api/analyses")
async def create_analysis(file: UploadFile = File(...), options: str = Form("{}"),
                          camera: UploadFile | None = File(None), speed_calib: UploadFile | None = File(None), u=Depends(user)):
    name = re.sub(r"[^\w.\-]+", "_", Path(file.filename or "video.mp4").name)[-80:]
    if not re.search(r"\.(mp4|mov|avi|mkv|m4v|webm)$", name, re.I):
        err("Upload a video file (mp4, mov, avi, mkv, webm)")
    try:
        o = json.loads(options)
    except Exception:
        err("Bad options")
    # every check runs on every video; speed is added below when a calibration file and a limit are given
    checks = [k for k in REGISTRY if k != "speed"]
    aid = db.new_id()
    work = config.ANALYSES / aid; (work / "original").mkdir(parents=True)
    dst = work / "original" / name
    size = 0
    with open(dst, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > config.MAX_UPLOAD_MB << 20:
                shutil.rmtree(work); err("File too large", 413)
            f.write(chunk)
    opts = {"checks": checks, "lanes": bool(o.get("lanes", True)), "plates": bool(o.get("plates", True))}
    if o.get("max_sec"):
        opts["max_sec"] = float(o["max_sec"])
    if camera is not None and camera.filename:
        p = work / "camera.json"; p.write_bytes(await camera.read()); json.loads(p.read_text()); opts["camera"] = str(p)
    if speed_calib is not None and speed_calib.filename and o.get("speed_limit"):
        p = work / "speed_calib.json"; p.write_bytes(await speed_calib.read()); json.loads(p.read_text())
        opts["speed_calib"] = str(p); opts["speed_limit"] = float(o["speed_limit"])
        if "speed" in REGISTRY:
            opts["checks"].append("speed")
    with db.conn() as c:
        c.execute("INSERT INTO analyses (id,user_id,filename,created,status,stage,progress,options) VALUES (?,?,?,?,?,?,?,?)",
                  (aid, u["id"], name, db.now(), "queued", "Queued", 0.0, json.dumps(opts)))
    pipeline.submit(aid)
    return {"id": aid}


@app.get("/api/analyses")
def list_analyses(u=Depends(user)):
    with db.conn() as c:
        rows = c.execute("""SELECT a.*, (SELECT COUNT(*) FROM findings f WHERE f.analysis_id=a.id) n_findings,
                            (SELECT COUNT(*) FROM findings f WHERE f.analysis_id=a.id AND f.status='pending_review') n_review,
                            (SELECT COUNT(*) FROM challans ch WHERE ch.analysis_id=a.id) n_challans
                            FROM analyses a ORDER BY created DESC LIMIT 200""").fetchall()
    return [db.row(r, ("options", "meta")) for r in rows]


@app.get("/api/analyses/{aid}")
def get_analysis(aid: str, u=Depends(user)):
    a = _analysis(aid)
    with db.conn() as c:
        a["findings"] = [db.row(r, db.FINDING_JSON) for r in c.execute(
            "SELECT * FROM findings WHERE analysis_id=? ORDER BY t", (aid,))]
        a["challans"] = [dict(r) for r in c.execute("SELECT * FROM challans WHERE analysis_id=?", (aid,))]
    return a


@app.delete("/api/analyses/{aid}")
def delete_analysis(aid: str, u=Depends(user)):
    """permanently delete a video analysis: stops it if still running, removes the video, evidence, findings,
    reviews and challans"""
    a = _analysis(aid)
    with db.conn() as c:
        c.execute("UPDATE analyses SET status='cancelled' WHERE id=?", (aid,))
    pipeline.kill_leftovers(aid)
    time.sleep(1.5)
    with db.conn() as c:
        fids = [r[0] for r in c.execute("SELECT id FROM findings WHERE analysis_id=?", (aid,))]
        n_ch = c.execute("SELECT COUNT(*) FROM challans WHERE analysis_id=?", (aid,)).fetchone()[0]
        c.executemany("DELETE FROM reviews WHERE finding_id=?", [(f,) for f in fids])
        c.execute("DELETE FROM challans WHERE analysis_id=?", (aid,))
        c.execute("DELETE FROM findings WHERE analysis_id=?", (aid,))
        c.execute("DELETE FROM analyses WHERE id=?", (aid,))
    shutil.rmtree(config.ANALYSES / aid, ignore_errors=True)
    pipeline.POOL.submit(pipeline.restore_after_delete, set(fids))   # merged detections elsewhere become cases again
    return {"ok": True, "deleted_findings": len(fids), "deleted_challans": n_ch}


@app.get("/api/analyses/{aid}/files/{path:path}")
def analysis_file(aid: str, path: str, u=Depends(user)):
    base = (config.ANALYSES / aid).resolve()
    p = (base / path).resolve()
    if not str(p).startswith(str(base)) or not p.is_file():
        err("Not found", 404)
    return FileResponse(p)


# ---------------- digital twin / plates ----------------
def _plates_json(aid):
    """plate readings of an analysis (the reader may have run in several batches: one per check)"""
    d = config.ANALYSES / aid / "plates"
    js = sorted(d.glob("**/*_plates.json")) if d.exists() else []
    if not js:
        return None
    out = {"vehicles": []}
    for j in js:
        V = json.loads(j.read_text())["vehicles"]
        for v in V:
            if v.get("evidence_crop") and not v["evidence_crop"].startswith("/"):
                v["evidence_crop"] = str(j.parent.relative_to(config.ANALYSES / aid) / v["evidence_crop"])
        out["vehicles"] += V
    return out


def _scene(aid):
    """background image for the twin: lane overlay if lanes were learned, else a mid-video frame"""
    a = _analysis(aid); art = (a.get("meta") or {}).get("artefacts", {})
    lanes = art.get("lanes", {})
    for k in ("background_lanes", "overlay"):
        if lanes.get(k) and (config.ANALYSES / aid / lanes[k]).exists():
            return lanes[k]
    p = config.ANALYSES / aid / "scene.jpg"
    if not p.exists():
        import cv2
        cap = cv2.VideoCapture(str(config.ANALYSES / aid / "playable.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2))
        ok, fr = cap.read()
        if ok:
            cv2.imwrite(str(p), fr)
    return "scene.jpg" if p.exists() else None


@app.get("/api/analyses/{aid}/twin")
def twin(aid: str, u=Depends(user)):
    a = get_analysis(aid, u)
    pj = _plates_json(aid)
    plates_by = {v["vehicle"]: v.get("plate") for v in (pj or {}).get("vehicles", []) if v["decision"] == "READ"}
    tracks, source = [], None
    cache = config.ANALYSES / aid / "det_cache.pkl"
    if cache.exists():            # every vehicle from the shared detection + tracking pass
        import pickle
        from collections import Counter, defaultdict
        with open(cache, "rb") as fh:
            c = pickle.load(fh)
        fps = (a.get("meta") or {}).get("fps") or 25
        names, VEH = c["names"], {"car", "bus", "truck", "autorickshaw", "motorcycle", "bicycle", "caravan", "trailer", "vehicle fallback"}
        path, types = defaultdict(list), defaultdict(Counter)
        for d in c["frames"]:
            if d["id"] is None:
                continue
            for b, k, i in zip(d["xyxy"].tolist(), d["cls"].astype(int).tolist(), d["id"].astype(int).tolist()):
                if names[k] in VEH:
                    path[i].append([round(d["fi"] / fps, 2)] + [round(v, 1) for v in b]); types[i][names[k]] += 1
        tracks = [{"id": i, "type": types[i].most_common(1)[0][0], "path": p, "plate": plates_by.get(i)} for i, p in path.items() if len(p) >= 2]
        source = "all vehicles from the shared detection + tracking pass (IDD detector + ByteTrack, 5 frames/s)"
    elif pj:                      # older analyses: tracks from the plate reader
        tracks = [{"id": v["vehicle"], "type": v["type"], "path": v.get("path", []), "plate": plates_by.get(v["vehicle"])}
                  for v in pj.get("vehicles", []) if v.get("path")]
        source = "all vehicles tracked by the plate tool (IDD detector + ByteTrack, 5 frames/s)"
    art = (a.get("meta") or {}).get("artefacts", {})
    return {"meta": a.get("meta"), "filename": a["filename"], "scene": _scene(aid), "tracks": tracks,
            "lanes": art.get("lanes_summary", {}), "lane_files": art.get("lanes", {}),
            "findings": [{k: f[k] for k in ("id", "violation", "t", "box", "route", "status", "track", "confidence")} for f in a["findings"]],
            "source": source if tracks else None}


class SignalSimReq(BaseModel):
    group: str = "toward"
    lanes: int = 3
    other: list[float]
    amb_per_hour: float = 4
    bus_stops: bool = True
    cattle_per_hour: float = 0
    minutes: float = 20


@app.get("/api/analyses/{aid}/signal-sim")
def signal_sim_get(aid: str, u=Depends(user)):
    """SIMULATED signal timing: traffic measured from this video, the state of the last simulation run"""
    a = _analysis(aid)
    fps = (a.get("meta") or {}).get("fps") or 25
    return {"ready": signal_sim.ready(), "measured": signal_sim.measure(aid, fps), **signal_sim.state(aid)}


@app.post("/api/analyses/{aid}/signal-sim")
def signal_sim_post(aid: str, req: SignalSimReq, u=Depends(user)):
    a = _analysis(aid)
    if a["status"] != "done":
        err("The analysis must finish first")
    if not signal_sim.ready():
        err("The simulator is not available on this server (Unity or signal_sim/ missing)", 503)
    if signal_sim.state(aid).get("status") in ("queued", "running"):
        err("A simulation is already running for this video", 409)
    try:
        signal_sim.start(aid, (a.get("meta") or {}).get("fps") or 25, req.model_dump())
    except ValueError as e:
        err(str(e))
    return {"ok": True}


@app.get("/api/plates")
def plates_all(u=Depends(user)):
    out = []
    with db.conn() as c:
        A = [dict(r) for r in c.execute("SELECT id, filename, created FROM analyses WHERE status='done' ORDER BY created DESC")]
        links = {}
        for r in c.execute("SELECT f.id, f.analysis_id, f.violation, f.evidence, ch.id cid, ch.number FROM findings f LEFT JOIN challans ch ON ch.finding_id=f.id"):
            pl = (json.loads(r["evidence"] or "{}").get("plate") or {})
            if pl.get("vehicle") is not None:
                links.setdefault((r["analysis_id"], str(pl["vehicle"])), []).append({"finding": r["id"], "violation": r["violation"], "challan": r["cid"], "number": r["number"]})
    for a in A:
        pj = _plates_json(a["id"])
        for v in (pj or {}).get("vehicles", []):
            crop = v.get("evidence_crop")
            out.append({"analysis_id": a["id"], "filename": a["filename"], "vehicle": v["vehicle"], "type": v["type"], "decision": v["decision"],
                        "plate": v.get("plate") if v["decision"] == "READ" else None, "confidence": v.get("confidence", 0), "reasons": v.get("reasons", []),
                        "t": v.get("first_seen_s"), "crop": crop or None, "rule": v.get("rule", "strict"),
                        "links": links.get((a["id"], str(v["vehicle"])), [])})
    return out


class Profile(BaseModel):
    name: str


class PwChange(BaseModel):
    current: str
    password: str


@app.post("/api/me")
def update_me(b: Profile, u=Depends(user)):
    if len(b.name.strip()) < 2:
        err("Enter your name")
    with db.conn() as c:
        c.execute("UPDATE users SET name=? WHERE id=?", (b.name.strip(), u["id"]))
    return {"ok": True}


@app.post("/api/me/password")
def change_pw(b: PwChange, request: Request, u=Depends(user)):
    if not sec.check_pw(b.current, u["pw_hash"]):
        err("Current password is incorrect")
    probs = sec.password_problems(b.password)
    if probs:
        err("Password needs " + ", ".join(probs))
    with db.conn() as c:
        c.execute("UPDATE users SET pw_hash=? WHERE id=?", (sec.hash_pw(b.password), u["id"]))
        c.execute("DELETE FROM sessions WHERE user_id=? AND token_hash<>?", (u["id"], sec._th(request.cookies.get(COOKIE, ""))))
    return {"ok": True}


@app.get("/api/analytics")
def analytics(u=Depends(user)):
    with db.conn() as c:
        A = [dict(r) for r in c.execute("SELECT id, filename, created, finished, status, meta FROM analyses")]
        rev = {r[0]: r[1] for r in c.execute("SELECT decision, COUNT(*) FROM reviews GROUP BY decision")}
        by_status = {r[0]: r[1] for r in c.execute("SELECT status, COUNT(*) FROM findings GROUP BY status")}
        by_det = {r[0]: r[1] for r in c.execute("SELECT detector, COUNT(*) FROM findings GROUP BY detector")}
    proc = []
    for a in A:
        if a["finished"] and a["status"] == "done":
            m = json.loads(a["meta"] or "{}")
            proc.append({"filename": a["filename"], "video_s": m.get("duration"), "processing_s": round(a["finished"] - a["created"], 1)})
    return {"reviews": rev, "by_status": by_status, "by_detector": by_det, "processing": proc, **stats(u)}


# ---------------- findings / review ----------------
class ReviewIn(BaseModel):
    decision: str        # challan | reject | more_evidence
    note: str = ""
    plate: str = ""      # optional: plate as read by the reviewer from the plate crop / frames
    offences: list[str] = []   # offences to charge when the case lists several for this vehicle (default: all)


@app.get("/api/findings")
def list_findings(status: str | None = None, u=Depends(user)):
    q, args = "SELECT f.*, a.filename FROM findings f JOIN analyses a ON a.id=f.analysis_id", []
    if status:
        q += " WHERE f.status=?"; args.append(status)
    with db.conn() as c:
        return [db.row(r, db.FINDING_JSON) for r in c.execute(q + " ORDER BY f.created DESC LIMIT 500", args)]


@app.get("/api/findings/{fid}")
def get_finding(fid: str, u=Depends(user)):
    with db.conn() as c:
        f = db.row(c.execute("SELECT f.*, a.filename FROM findings f JOIN analyses a ON a.id=f.analysis_id WHERE f.id=?", (fid,)).fetchone(), db.FINDING_JSON)
        if not f:
            err("Not found", 404)
        f["reviews"] = [dict(r) | {"reviewer": (c.execute("SELECT name FROM users WHERE id=?", (r["user_id"],)).fetchone() or {"name": "?"})["name"]}
                        for r in c.execute("SELECT * FROM reviews WHERE finding_id=? ORDER BY created", (fid,))]
        ch = c.execute("SELECT * FROM challans WHERE finding_id=?", (fid,)).fetchone()
        f["challan"] = db.row(ch, ("offences",)) if ch else None
        f["merged"] = _merged(c, fid)
        f["merged_into"] = db.row(c.execute("SELECT f.id, f.violation, f.status, f.t, f.analysis_id, a.filename FROM findings f "
                                            "JOIN analyses a ON a.id=f.analysis_id WHERE f.id=?", (f.get("duplicate_of"),)).fetchone()) \
            if f.get("duplicate_of") else None
    return f


def _merged(c, fid):
    """every other detection of the same vehicle merged into this case, with its evidence (one case, all evidence)"""
    p = db.row(c.execute("SELECT * FROM findings WHERE id=?", (fid,)).fetchone())
    out = [db.row(r, db.FINDING_JSON) for r in c.execute(
        "SELECT f.*, a.filename FROM findings f JOIN analyses a ON a.id=f.analysis_id WHERE f.duplicate_of=? ORDER BY a.created, f.t", (fid,))]
    for g in out:
        g["why"] = _why(g, p)
    return out


def _why(g, p):
    """why a merged detection belongs to case p, told from that case"""
    if g["analysis_id"] != p["analysis_id"]:
        return f"Same vehicle at the same moment, in another upload of this footage ({g['filename']})"
    if g["violation"] != p["violation"]:
        return f"Same vehicle: {g['violation'].lower()} was detected as well"
    raw = next((r for r in reversed(g.get("uncertain_reasons") or []) if r.startswith("duplicate")), "")
    a, b = re.match(r"H(\d+)\.(\d+)", g["track"] or ""), re.match(r"H(\d+)\.(\d+)", p["track"] or "")
    if a and b and a.group(1) == b.group(1) and a.group(2) != b.group(2):
        return "Another person on this vehicle (rider / pillion), also without a helmet"
    if "image similarity" in raw:
        return "Same-looking " + raw.split("same-looking ", 1)[-1]
    if "number plate" in raw:
        return raw.split(": ", 1)[-1].capitalize()
    return "The same vehicle, seen again a moment later"


@app.post("/api/findings/{fid}/review")
def review(fid: str, b: ReviewIn, u=Depends(user)):
    if b.decision not in ("challan", "reject", "more_evidence"):
        err("Unknown decision")
    with db.conn() as c:
        f = c.execute("SELECT * FROM findings WHERE id=?", (fid,)).fetchone()
        if not f:
            err("Not found", 404)
        if f["status"] not in ("pending_review", "needs_evidence"):
            err("This case was already decided")
        c.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?)", (db.new_id(), fid, u["id"], b.decision, b.note[:1000], db.now()))
        new = {"challan": "challan", "reject": "rejected", "more_evidence": "needs_evidence"}[b.decision]
        c.execute("UPDATE findings SET status=? WHERE id=?", (new, fid))
        aid = f["analysis_id"]
    pl = re.sub(r"[^A-Z0-9 ]", "", b.plate.upper()).strip()[:14] or None
    cid = pipeline.create_challan(fid, aid, "human_review", u["id"], reviewer_plate=pl, offences=b.offences or None) \
        if b.decision == "challan" else None
    return {"ok": True, "challan_id": cid}


# ---------------- challans ----------------
@app.get("/api/challans")
def list_challans(u=Depends(user)):
    with db.conn() as c:
        rows = c.execute("""SELECT ch.*, f.violation, f.confidence, f.t, f.track, f.vehicle, f.evidence, a.filename
                            FROM challans ch JOIN findings f ON f.id=ch.finding_id JOIN analyses a ON a.id=ch.analysis_id
                            ORDER BY ch.created DESC""").fetchall()
    return [db.row(r, ("evidence", "offences")) for r in rows]


def _challan_bundle(cid):
    with db.conn() as c:
        ch = db.row(c.execute("SELECT * FROM challans WHERE id=?", (cid,)).fetchone(), ("offences",))
        if not ch:
            err("Not found", 404)
        f = db.row(c.execute("SELECT * FROM findings WHERE id=?", (ch["finding_id"],)).fetchone(), db.FINDING_JSON)
        a = db.row(c.execute("SELECT * FROM analyses WHERE id=?", (ch["analysis_id"],)).fetchone(), ("options", "meta"))
        rv = db.row(c.execute("SELECT * FROM reviews WHERE finding_id=? AND decision='challan' ORDER BY created DESC", (f["id"],)).fetchone())
        who = db.row(c.execute("SELECT id,name FROM users WHERE id=?", (rv["user_id"],)).fetchone()) if rv else None
    return ch, f, a, rv, who


@app.get("/api/challans/{cid}")
def get_challan(cid: str, u=Depends(user)):
    ch, f, a, rv, who = _challan_bundle(cid)
    with db.conn() as c:
        f["merged"] = _merged(c, f["id"])
    return {"challan": ch, "finding": f, "analysis": {"id": a["id"], "filename": a["filename"], "created": a["created"]},
            "review": rv, "reviewer": who}


@app.get("/api/challans/{cid}/pdf")
def challan_pdf_route(cid: str, u=Depends(user)):
    ch, f, a, rv, who = _challan_bundle(cid)
    out = config.ANALYSES / a["id"] / "challans"; out.mkdir(exist_ok=True)
    p = out / f"{ch['number']}.pdf"
    with db.conn() as c:                         # every other detection of this vehicle: its evidence goes in too
        merged = _merged(c, f["id"])
    offs = {o["id"]: o for o in f.get("offences") or []}
    extra = [(g, config.ANALYSES / g["analysis_id"] / "evidence" / g["id"], offs.get(g["id"])) for g in merged]
    challan_pdf.build(p, ch, f, a, rv, who, config.ANALYSES / a["id"] / "evidence" / f["id"], extra)
    return FileResponse(p, media_type="application/pdf", filename=f"{ch['number']}.pdf")


# ---------------- statistics (real counts only) ----------------
@app.get("/api/stats")
def stats(u=Depends(user)):
    with db.conn() as c:
        one = lambda q, *a: c.execute(q, a).fetchone()[0]
        by_type = {r[0]: r[1] for r in c.execute("SELECT violation, COUNT(*) FROM findings GROUP BY violation")}
        conf = [r[0] for r in c.execute("SELECT confidence FROM findings")]
        return {"analyses": one("SELECT COUNT(*) FROM analyses"),
                "analyses_done": one("SELECT COUNT(*) FROM analyses WHERE status='done'"),
                "processing": one("SELECT COUNT(*) FROM analyses WHERE status IN ('queued','processing')"),
                "findings": one("SELECT COUNT(*) FROM findings"),
                "confirmed": one("SELECT COUNT(*) FROM findings WHERE route='confirmed'"),
                "pending_review": one("SELECT COUNT(*) FROM findings WHERE status='pending_review'"),
                "needs_evidence": one("SELECT COUNT(*) FROM findings WHERE status='needs_evidence'"),
                "rejected": one("SELECT COUNT(*) FROM findings WHERE status='rejected'"),
                "exempt": one("SELECT COUNT(*) FROM findings WHERE status='exempt'"),
                "challans": one("SELECT COUNT(*) FROM challans"),
                "challans_reviewed": one("SELECT COUNT(*) FROM challans WHERE source='human_review'"),
                "by_type": by_type,
                "confidence_hist": [sum(1 for x in conf if i / 10 <= x < (i + 1) / 10 or (i == 9 and x == 1)) for i in range(10)]}


# ---------------- 3-D simulator (Unity WebGL build, separate from the React app) ----------------
SIM3D = config.ROOT / "signal_sim" / "Builds" / "WebGL"
SIM3D_TYPES = {".wasm": "application/wasm", ".js": "application/javascript", ".data": "application/octet-stream",
               ".html": "text/html", ".css": "text/css", ".png": "image/png", ".ico": "image/x-icon"}


@app.get("/sim3d")
def sim3d_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/sim3d/")


@app.get("/sim3d/{path:path}")
def sim3d(path: str):
    p = (SIM3D / (path or "index.html")).resolve()
    if str(p).startswith(str(SIM3D.resolve())) and p.is_file():
        return FileResponse(p, media_type=SIM3D_TYPES.get(p.suffix.lower()))
    if not (SIM3D / "index.html").exists():
        from fastapi.responses import HTMLResponse
        return HTMLResponse("<h2>3-D simulator not built yet</h2><p>Build it with Unity: "
                            "-executeMethod SignalSim.WebBuild.Run (needs WebGL Build Support).</p>", 503)
    return JSONResponse({"detail": "not found"}, 404)


# ---------------- web app ----------------
@app.get("/{path:path}")
def spa(path: str):
    p = (config.WEB_DIST / path).resolve()
    if path and str(p).startswith(str(config.WEB_DIST.resolve())) and p.is_file():
        return FileResponse(p)
    idx = config.WEB_DIST / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return JSONResponse({"detail": "web app not built - run: cd app/web && npm run build"}, 503)
