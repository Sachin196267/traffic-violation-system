"""
SQLite database initialization and query helpers.
Single-file DB at webapp/traffic_violations.db.
"""
from __future__ import annotations
import os
import sqlite3
import json
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "traffic_violations.db")

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    """Create tables if they don't exist and run incremental column migrations."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            status TEXT DEFAULT 'queued',
            input_type TEXT,
            input_source TEXT,
            video_path TEXT,
            output_video_path TEXT,
            max_frames INTEGER DEFAULT 150,
            start_frame INTEGER DEFAULT 0,
            total_challans INTEGER DEFAULT 0,
            progress INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS challans (
            id TEXT PRIMARY KEY,
            job_id TEXT REFERENCES jobs(id),
            frame_idx INTEGER,
            track_id INTEGER,
            vehicle_class TEXT,
            plate_number TEXT,
            violations TEXT,
            fine_inr INTEGER,
            severity_level TEXT,
            severity_score INTEGER,
            speed_kmh REAL,
            timestamp_str TEXT,
            lane TEXT,
            xai_data TEXT,
            pdf_path TEXT,
            scene_img_path TEXT,
            vehicle_img_path TEXT,
            plate_img_path TEXT,
            gradcam_img_path TEXT
        );
    """)

    # Check for missing columns on existing tables and add them gracefully
    challan_cols = [c["name"] for c in conn.execute("PRAGMA table_info(challans)").fetchall()]
    desired_cols = {
        "speed_kmh": "REAL",
        "timestamp_str": "TEXT",
        "lane": "TEXT",
        "xai_data": "TEXT"
    }
    for col, col_type in desired_cols.items():
        if col not in challan_cols:
            conn.execute(f"ALTER TABLE challans ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()


# ── User queries ──────────────────────────────────────────────

def create_user(email: str, password_hash: str) -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
        (email, password_hash),
    )
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return uid

def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return row

def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return row


# ── Job queries ───────────────────────────────────────────────

def create_job(job_id: str, user_id: int, input_type: str, input_source: str,
               video_path: str, max_frames: int, start_frame: int) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO jobs (id, user_id, status, input_type, input_source,
           video_path, max_frames, start_frame)
           VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)""",
        (job_id, user_id, input_type, input_source, video_path, max_frames, start_frame),
    )
    conn.commit()
    conn.close()

def update_job_status(job_id: str, status: str, **kwargs):
    conn = get_connection()
    sets = ["status = ?"]
    vals = [status]
    for k, v in kwargs.items():
        sets.append(f"{k} = ?")
        vals.append(v)
    vals.append(job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)
    conn.commit()
    conn.close()

def get_job(job_id: str) -> Optional[sqlite3.Row]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return row

def get_user_jobs(user_id: int) -> list[sqlite3.Row]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
    ).fetchall()
    conn.close()
    return rows


# ── Challan queries ───────────────────────────────────────────

def insert_challan(challan: dict, job_id: str) -> None:
    conn = get_connection()
    xai_raw = challan.get("xai_data")
    xai_str = json.dumps(xai_raw) if isinstance(xai_raw, (dict, list)) else (xai_raw or "")

    conn.execute(
        """INSERT OR REPLACE INTO challans
           (id, job_id, frame_idx, track_id, vehicle_class, plate_number,
            violations, fine_inr, severity_level, severity_score,
            speed_kmh, timestamp_str, lane, xai_data,
            pdf_path, scene_img_path, vehicle_img_path, plate_img_path, gradcam_img_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            challan["challan_id"], job_id, challan.get("frame"),
            challan.get("track_id"), challan.get("class"), challan.get("plate"),
            json.dumps(challan.get("violations", [])), challan.get("fine_inr"),
            challan.get("severity"), challan.get("score"),
            challan.get("speed_kmh", 0.0), challan.get("timestamp_str", ""),
            challan.get("lane", ""), xai_str,
            challan.get("pdf"), challan.get("scene_img"),
            challan.get("vehicle_img"), challan.get("plate_img"),
            challan.get("gradcam_img"),
        ),
    )
    conn.commit()
    conn.close()

def get_job_challans(job_id: str) -> list[sqlite3.Row]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM challans WHERE job_id = ? ORDER BY frame_idx", (job_id,)
    ).fetchall()
    conn.close()
    return rows

def get_challan(challan_id: str) -> Optional[sqlite3.Row]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM challans WHERE id = ?", (challan_id,)).fetchone()
    conn.close()
    return row
