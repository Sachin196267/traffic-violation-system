"""SQLite storage (stdlib only). One connection per call; WAL mode for concurrent reads during processing."""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE, phone TEXT UNIQUE,
  pw_hash TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0, role TEXT NOT NULL DEFAULT 'operator',
  created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS codes (
  id TEXT PRIMARY KEY, target TEXT NOT NULL, purpose TEXT NOT NULL, code_hash TEXT NOT NULL,
  expires REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, used INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS analyses (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, filename TEXT NOT NULL, created REAL NOT NULL,
  status TEXT NOT NULL, stage TEXT, progress REAL NOT NULL DEFAULT 0, error TEXT,
  options TEXT NOT NULL, meta TEXT, finished REAL);
CREATE TABLE IF NOT EXISTS findings (
  id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL, detector TEXT NOT NULL, violation TEXT NOT NULL,
  track TEXT, t REAL, t_start REAL, t_end REAL, confidence REAL, model_decision TEXT NOT NULL,
  route TEXT NOT NULL, status TEXT NOT NULL, box TEXT, reasons TEXT, uncertain_reasons TEXT,
  evidence TEXT, emergency TEXT, vehicle TEXT, model_info TEXT, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (
  id TEXT PRIMARY KEY, finding_id TEXT NOT NULL, user_id TEXT NOT NULL, decision TEXT NOT NULL,
  note TEXT, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS challans (
  id TEXT PRIMARY KEY, number TEXT UNIQUE NOT NULL, finding_id TEXT NOT NULL, analysis_id TEXT NOT NULL,
  created REAL NOT NULL, source TEXT NOT NULL, reviewer_id TEXT, status TEXT NOT NULL,
  plate TEXT, evidence_hash TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS f_analysis ON findings(analysis_id);
CREATE UNIQUE INDEX IF NOT EXISTS ch_finding ON challans(finding_id);
CREATE INDEX IF NOT EXISTS f_status ON findings(status);
"""


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)
        cols = [r[1] for r in c.execute("PRAGMA table_info(findings)")]
        if "duplicate_of" not in cols:
            c.execute("ALTER TABLE findings ADD COLUMN duplicate_of TEXT")
        if "offences" not in cols:       # other offences of the same vehicle, merged into this one case
            c.execute("ALTER TABLE findings ADD COLUMN offences TEXT")
        if "offences" not in [r[1] for r in c.execute("PRAGMA table_info(challans)")]:
            c.execute("ALTER TABLE challans ADD COLUMN offences TEXT")      # offences charged on this challan


def new_id():
    return uuid.uuid4().hex[:16]


def row(r, json_fields=()):
    if r is None:
        return None
    d = dict(r)
    for k in json_fields:
        if d.get(k):
            d[k] = json.loads(d[k])
    return d


FINDING_JSON = ("box", "reasons", "uncertain_reasons", "evidence", "model_info", "offences")


def now():
    return time.time()
