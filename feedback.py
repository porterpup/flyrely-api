"""In-app beta feedback.

POST /feedback  - any signed-in user (Supabase bearer token); stored in SQLite next to the prediction log
                  (FLYRELY_FEEDBACK_DB, default: same directory as FLYRELY_PREDICTION_DB), rate limited per user.
GET  /feedback  - admin only (same check as /usage); newest first.

If FLYRELY_NTFY_TOPIC is set, a short push goes to https://ntfy.sh/<topic>. It is fire-and-forget:
failures are logged and never affect the response.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

import admin_auth

logger = logging.getLogger("flyrely.feedback")

RATE_LIMIT = 10
RATE_WINDOW_S = 3600
MAX_CONTEXT_BYTES = 8000

router = APIRouter()

_DDL = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    email TEXT NOT NULL,
    category TEXT,
    message TEXT NOT NULL,
    context_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_email_ts ON feedback (email, ts_utc);
"""

_lock = threading.Lock()
_ready_paths: set[str] = set()


def db_path() -> Path:
    explicit = os.getenv("FLYRELY_FEEDBACK_DB")
    if explicit:
        return Path(explicit)
    pred = Path(os.getenv("FLYRELY_PREDICTION_DB", "./data/predictions.sqlite"))
    return pred.parent / "feedback.sqlite"


def _connect() -> sqlite3.Connection:
    path = db_path()
    if str(path) not in _ready_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=5)) as conn:
            conn.executescript(_DDL)
        _ready_paths.add(str(path))
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


class FlightContext(BaseModel):
    model_config = ConfigDict(extra="ignore")
    airline: Optional[str] = Field(default=None, max_length=64)
    flight_number: Optional[str] = Field(default=None, max_length=16)
    origin: Optional[str] = Field(default=None, max_length=8)
    destination: Optional[str] = Field(default=None, max_length=8)
    departure_time: Optional[str] = Field(default=None, max_length=64)
    delay_probability: Optional[float] = None
    risk_level: Optional[str] = Field(default=None, max_length=32)


class FeedbackContext(BaseModel):
    model_config = ConfigDict(extra="ignore")
    route: Optional[str] = Field(default=None, max_length=300)
    flight: Optional[FlightContext] = None
    app_version: Optional[str] = Field(default=None, max_length=64)
    user_agent: Optional[str] = Field(default=None, max_length=400)


class FeedbackIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    category: Optional[Literal["bug", "wrong prediction", "idea", "other"]] = None
    context: Optional[FeedbackContext] = None


async def require_user(authorization: Optional[str] = Header(default=None)) -> str:
    """Email of the signed-in Supabase user, or 401."""
    token = admin_auth._bearer(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail="Sign in required", headers={"WWW-Authenticate": "Bearer"})
    return await admin_auth._lookup_email(token)


def _rate_limited(conn: sqlite3.Connection, email: str, now: datetime) -> bool:
    cutoff = datetime.fromtimestamp(now.timestamp() - RATE_WINDOW_S, tz=timezone.utc).isoformat()
    n = conn.execute("SELECT COUNT(*) FROM feedback WHERE email = ? AND ts_utc > ?", (email, cutoff)).fetchone()[0]
    return n >= RATE_LIMIT


def _notify(email: str, message: str) -> None:
    topic = os.getenv("FLYRELY_NTFY_TOPIC", "").strip()
    if not topic:
        return
    snippet = " ".join(message.split())[:100]
    try:
        httpx.post(f"https://ntfy.sh/{topic}", content=f"New feedback from {email}: {snippet}".encode(),
                   headers={"Title": "FlyRely feedback"}, timeout=5.0)
    except Exception as e:  # never let a push failure surface
        logger.warning("ntfy notification failed: %s: %s", type(e).__name__, e)


@router.post("/feedback", status_code=201)
async def post_feedback(body: FeedbackIn, background: BackgroundTasks, email: str = Depends(require_user)):
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message is empty")
    ctx = body.context.model_dump(exclude_none=True) if body.context else None
    ctx_json = json.dumps(ctx) if ctx else None
    if ctx_json and len(ctx_json) > MAX_CONTEXT_BYTES:
        ctx_json = None
    now = datetime.now(timezone.utc)
    with _lock, closing(_connect()) as conn:
        if _rate_limited(conn, email, now):
            raise HTTPException(status_code=429, detail="Too many feedback messages; try again later",
                                headers={"Retry-After": str(RATE_WINDOW_S)})
        with conn:
            cur = conn.execute(
                "INSERT INTO feedback (ts_utc, email, category, message, context_json) VALUES (?, ?, ?, ?, ?)",
                (now.isoformat(), email, body.category, message, ctx_json))
        fid = cur.lastrowid
    background.add_task(_notify, email, message)
    return {"id": fid, "status": "received"}


@router.get("/feedback")
async def list_feedback(limit: int = Query(default=100, ge=1, le=500), _admin: str = Depends(admin_auth.require_admin)):
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    items = []
    for r in rows:
        d = dict(r)
        d["context"] = json.loads(d.pop("context_json")) if d["context_json"] else None
        items.append(d)
    return {"count": len(items), "items": items}
