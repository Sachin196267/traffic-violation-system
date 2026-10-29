"""
JWT Authentication: register, login, token validation, FastAPI dependency.
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt

# Ensure passlib works reliably with modern bcrypt >= 4.0.0
if not hasattr(bcrypt, "__about__"):
    class _BcryptAbout:
        __version__ = getattr(bcrypt, "__version__", "5.0.0")
    bcrypt.__about__ = _BcryptAbout()

import passlib.handlers.bcrypt
def _patched_finalize(cls, name, dryrun=False):
    return True
passlib.handlers.bcrypt._BcryptCommon._finalize_backend_mixin = classmethod(_patched_finalize)

from passlib.context import CryptContext
import jwt

from database import get_user_by_email, get_user_by_id, create_user

SECRET_KEY = os.environ.get("JWT_SECRET", "tvs-webapp-secret-change-in-prod-2026")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        # Fallback to direct bcrypt check if hash scheme varies
        try:
            return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False


def create_token(user_id: int, email: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """FastAPI dependency: extracts and validates JWT, returns user dict."""
    payload = decode_token(creds.credentials)
    try:
        uid = int(payload["sub"])
    except (ValueError, TypeError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid user identifier in token")

    user = get_user_by_id(uid)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"id": user["id"], "email": user["email"]}


async def get_current_user_flexible(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security_optional),
    token: Optional[str] = None,
) -> dict:
    """
    Flexible auth dependency:
    Accepts JWT from Authorization: Bearer header, ?token= query parameter, or tvs_token cookie.
    Ensures HTML5 video players, image tags, and direct download links work reliably.
    """
    raw_token = None
    if creds and creds.credentials:
        raw_token = creds.credentials
    elif token:
        raw_token = token
    elif "tvs_token" in request.cookies:
        raw_token = request.cookies.get("tvs_token")

    if not raw_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_token(raw_token)
    try:
        uid = int(payload["sub"])
    except (ValueError, TypeError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid user identifier in token")

    user = get_user_by_id(uid)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"id": user["id"], "email": user["email"]}


def register_user(email: str, password: str) -> dict:
    existing = get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    pw_hash = hash_password(password)
    uid = create_user(email, pw_hash)
    token = create_token(uid, email)
    return {"user_id": uid, "email": email, "token": token}


def login_user(email: str, password: str) -> dict:
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(user["id"], user["email"])
    return {"user_id": user["id"], "email": user["email"], "token": token}
