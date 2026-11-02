"""
Traffic Violation System — FastAPI Web Application.
Entry point: uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import os
import sys
import json
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# Ensure webapp dir is on path for local imports
WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
if WEBAPP_DIR not in sys.path:
    sys.path.insert(0, WEBAPP_DIR)

from database import (
    init_db, create_job, get_job, get_user_jobs,
    get_job_challans, get_challan, update_job_status
)
from auth import get_current_user, get_current_user_flexible, register_user, login_user
from models import AuthRequest, AuthResponse, JobURLRequest, JobStatusResponse, ChallanResponse
from jobs import submit_job, submit_url_job, generate_job_id
from pipeline_runner import job_progress

PROJECT_ROOT = os.path.dirname(WEBAPP_DIR)

app = FastAPI(title="Traffic Violation Enforcement System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Exception Handlers (ALWAYS return JSON, never HTML) ───

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Formulate friendly error string
    errors = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        errors.append(f"{loc}: {msg}" if loc else msg)
    friendly_msg = "; ".join(errors) if errors else "Validation error"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": friendly_msg, "errors": exc.errors()},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"},
    )

# Serve static files
app.mount("/static", StaticFiles(directory=os.path.join(WEBAPP_DIR, "static")), name="static")

# Initialize DB on startup
@app.on_event("startup")
def startup():
    init_db()


# ── HTML Page Routes ──────────────────────────────────────────

def _serve_template(name: str) -> HTMLResponse:
    path = os.path.join(WEBAPP_DIR, "templates", name)
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/", response_class=HTMLResponse)
def page_login():
    return _serve_template("login.html")

@app.get("/register", response_class=HTMLResponse)
def page_register():
    return _serve_template("register.html")

@app.get("/dashboard", response_class=HTMLResponse)
def page_dashboard():
    return _serve_template("dashboard.html")

@app.get("/results", response_class=HTMLResponse)
def page_results():
    return _serve_template("results.html")

@app.get("/history", response_class=HTMLResponse)
def page_history():
    return _serve_template("history.html")


# ── Auth API ──────────────────────────────────────────────────

@app.post("/api/auth/register")
def api_register(req: AuthRequest, response: Response):
    data = register_user(req.email, req.password)
    response.set_cookie(
        key="tvs_token",
        value=data["token"],
        path="/",
        max_age=86400,
        httponly=False,
        samesite="lax"
    )
    return data

@app.post("/api/auth/login")
def api_login(req: AuthRequest, response: Response):
    data = login_user(req.email, req.password)
    response.set_cookie(
        key="tvs_token",
        value=data["token"],
        path="/",
        max_age=86400,
        httponly=False,
        samesite="lax"
    )
    return data

@app.get("/api/auth/me")
def api_me(user=Depends(get_current_user)):
    return {"id": user["id"], "email": user["email"]}


# ── Job API ───────────────────────────────────────────────────

@app.post("/api/jobs/upload")
async def api_upload_video(
    file: UploadFile = File(...),
    max_frames: int = Form(default=300),
    start_frame: int = Form(default=0),
    user=Depends(get_current_user),
):
    if max_frames < 1 or max_frames > 100000:
        raise HTTPException(400, "max_frames must be between 1 and 100,000")

    job_id = generate_job_id()
    job_dir = os.path.join(WEBAPP_DIR, "jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save uploaded video
    video_path = os.path.join(job_dir, file.filename)
    with open(video_path, "wb") as out:
        content = await file.read()
        out.write(content)

    create_job(job_id, user["id"], "upload", file.filename, video_path, max_frames, start_frame)
    submit_job(job_id, video_path, max_frames, start_frame)

    return {"job_id": job_id, "status": "queued"}


@app.post("/api/jobs/url")
def api_submit_url(
    req: JobURLRequest,
    user=Depends(get_current_user),
):
    if req.max_frames < 1 or req.max_frames > 100000:
        raise HTTPException(400, "max_frames must be between 1 and 100,000")

    job_id = generate_job_id()
    job_dir = os.path.join(WEBAPP_DIR, "jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)

    video_path = os.path.join(job_dir, "input_video.mp4")

    create_job(job_id, user["id"], "url", req.url, video_path, req.max_frames, req.start_frame)
    submit_url_job(job_id, req.url, video_path, req.max_frames, req.start_frame)

    return {"job_id": job_id, "status": "downloading"}


@app.get("/api/jobs")
def api_list_jobs(user=Depends(get_current_user)):
    rows = get_user_jobs(user["id"])
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "status": r["status"],
            "input_type": r["input_type"],
            "input_source": r["input_source"],
            "max_frames": r["max_frames"],
            "start_frame": r["start_frame"],
            "total_challans": r["total_challans"],
            "progress": job_progress.get(r["id"], r["progress"]),
            "error_message": r["error_message"],
            "created_at": r["created_at"],
            "completed_at": r["completed_at"],
        })
    return result


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str, user=Depends(get_current_user)):
    job = get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    return {
        "id": job["id"],
        "status": job["status"],
        "input_type": job["input_type"],
        "input_source": job["input_source"],
        "max_frames": job["max_frames"],
        "start_frame": job["start_frame"],
        "total_challans": job["total_challans"],
        "progress": job_progress.get(job_id, job["progress"]),
        "error_message": job["error_message"],
        "created_at": job["created_at"],
        "completed_at": job["completed_at"],
    }


@app.get("/api/jobs/{job_id}/challans")
def api_get_challans(job_id: str, user=Depends(get_current_user)):
    job = get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    rows = get_job_challans(job_id)
    result = []
    for r in rows:
        xai_val = r["xai_data"] if "xai_data" in r.keys() else None
        if isinstance(xai_val, str) and xai_val.strip().startswith(("{", "[")):
            try:
                xai_val = json.loads(xai_val)
            except Exception:
                pass

        result.append({
            "id": r["id"],
            "job_id": r["job_id"],
            "frame_idx": r["frame_idx"],
            "track_id": r["track_id"],
            "vehicle_class": r["vehicle_class"],
            "plate_number": r["plate_number"],
            "violations": json.loads(r["violations"]) if r["violations"] else [],
            "severity_level": r["severity_level"],
            "severity_score": r["severity_score"],
            "speed_kmh": r["speed_kmh"] if "speed_kmh" in r.keys() else 0.0,
            "timestamp_str": r["timestamp_str"] if "timestamp_str" in r.keys() else "",
            "lane": r["lane"] if "lane" in r.keys() else "",
            "xai_data": xai_val,
        })
    return result


# ── File Serving (supports GET & HEAD, authenticated via Bearer/Query/Cookie) ───

@app.api_route("/api/files/{job_id}/video", methods=["GET", "HEAD"])
def api_serve_video(job_id: str, user=Depends(get_current_user_flexible)):
    job = get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    video_path = os.path.join(WEBAPP_DIR, "jobs", job_id, "processed_output.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(404, "Output video not yet available")
    return FileResponse(video_path, media_type="video/mp4", filename="processed_output.mp4")


@app.api_route("/api/files/{job_id}/twin", methods=["GET", "HEAD"])
def api_serve_twin(job_id: str, user=Depends(get_current_user_flexible)):
    job = get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")

    twin_video = os.path.join(WEBAPP_DIR, "jobs", job_id, "mini_digital_twin.mp4")
    if os.path.exists(twin_video):
        return FileResponse(twin_video, media_type="video/mp4", filename="mini_digital_twin.mp4")

    twin_preview = os.path.join(WEBAPP_DIR, "jobs", job_id, "mini_digital_twin_preview.jpg")
    if os.path.exists(twin_preview):
        return FileResponse(twin_preview, media_type="image/jpeg", filename="mini_digital_twin_preview.jpg")

    raise HTTPException(404, "Mini digital twin not available")


@app.api_route("/api/files/{job_id}/lane_result", methods=["GET", "HEAD"])
def api_serve_lane_result(job_id: str, user=Depends(get_current_user_flexible)):
    job = get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    lane_path = os.path.join(WEBAPP_DIR, "jobs", job_id, "lane_detection_result.jpg")
    if not os.path.exists(lane_path):
        raise HTTPException(404, "Lane detection result not available")
    return FileResponse(lane_path, media_type="image/jpeg", filename="lane_detection_result.jpg")


@app.api_route("/api/files/{job_id}/evidence/{filename}", methods=["GET", "HEAD"])
def api_serve_evidence(job_id: str, filename: str, user=Depends(get_current_user_flexible)):
    job = get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    file_path = os.path.join(WEBAPP_DIR, "jobs", job_id, "evidence", filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "Evidence file not found")
    media = "image/jpeg" if filename.endswith((".jpg", ".jpeg")) else "image/png"
    return FileResponse(file_path, media_type=media)


@app.api_route("/api/files/{job_id}/challans/{filename}", methods=["GET", "HEAD"])
def api_serve_challan(job_id: str, filename: str, user=Depends(get_current_user_flexible)):
    job = get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    file_path = os.path.join(WEBAPP_DIR, "jobs", job_id, "challans", filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "Challan PDF not found")
    return FileResponse(file_path, media_type="application/pdf", filename=filename)


@app.api_route("/api/files/{job_id}/report", methods=["GET", "HEAD"])
def api_serve_session_report(job_id: str, user=Depends(get_current_user_flexible)):
    job = get_job(job_id)
    if not job or job["user_id"] != user["id"]:
        raise HTTPException(404, "Job not found")
    report_path = os.path.join(WEBAPP_DIR, "jobs", job_id, "session_enforcement_report.pdf")
    if not os.path.exists(report_path):
        raise HTTPException(404, "Session enforcement report not found")
    return FileResponse(report_path, media_type="application/pdf", filename=f"ITMS_Enforcement_Report_{job_id}.pdf")
