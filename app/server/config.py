"""Paths and settings for the traffic-violation platform backend."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # project root (models, tools)
APP = ROOT / "app"
DATA = Path(os.environ.get("TVP_DATA", APP / "data"))
ANALYSES = DATA / "analyses"
DB_PATH = DATA / "platform.db"
WEB_DIST = APP / "web" / "dist"
PY = os.environ.get("TVP_PYTHON", str(ROOT / ".venv/bin/python"))

# Verification codes are delivered by a notifier. Without SMTP/SMS credentials the platform runs in
# developer mode: codes are written to DATA/dev_outbox.log and shown in a clearly marked dev banner.
SMTP_HOST = os.environ.get("TVP_SMTP_HOST")
SMTP_USER = os.environ.get("TVP_SMTP_USER")
SMTP_PASS = os.environ.get("TVP_SMTP_PASS")
SMTP_FROM = os.environ.get("TVP_SMTP_FROM", SMTP_USER or "noreply@localhost")
DEV_MODE = not SMTP_HOST

SESSION_DAYS = 7
MAX_UPLOAD_MB = 2048

for d in (DATA, ANALYSES):
    d.mkdir(parents=True, exist_ok=True)
