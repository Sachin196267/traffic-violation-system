"""
Pydantic schemas for API request/response validation.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Any


class AuthRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=6)


class AuthResponse(BaseModel):
    user_id: int
    email: str
    token: str


class UserInfo(BaseModel):
    id: int
    email: str


class JobURLRequest(BaseModel):
    url: str
    max_frames: int = Field(default=300, ge=1, le=100000)
    start_frame: int = Field(default=0, ge=0)


class JobStatusResponse(BaseModel):
    id: str
    status: str
    input_type: Optional[str] = None
    input_source: Optional[str] = None
    max_frames: int = 150
    start_frame: int = 0
    total_challans: int = 0
    progress: int = 0
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class ChallanResponse(BaseModel):
    id: str
    job_id: str
    frame_idx: Optional[int] = None
    track_id: Optional[int] = None
    vehicle_class: Optional[str] = None
    plate_number: Optional[str] = None
    violations: Optional[list[str]] = None
    severity_level: Optional[str] = None
    severity_score: Optional[int] = None
    speed_kmh: Optional[float] = None
    timestamp_str: Optional[str] = None
    lane: Optional[str] = None
    xai_data: Optional[Any] = None
