"""'Were you delayed?' self-reports: one answer per signed-in user per flight.

POST /flight-reports  - any signed-in user (Supabase bearer token). `answer` saves or edits the user's single row for
                        that flight; `answer: null` withdraws it (Undo). Rate limited per user.
GET  /flight-reports  - admin only (same check as /usage); newest first.

Stored in SQLite next to the prediction log (FLYRELY_FLIGHT_REPORTS_DB, default: flight_reports.sqlite in the
directory of FLYRELY_PREDICTION_DB). The user id (Supabase uuid) is stored; the email, user agent and IP never are.
`tester` marks admin / @example.com accounts so the scorer can leave them out.
"""

import logging
import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional, Tuple

from fastapi import FastAPI, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

import admin_auth
import prediction_log

logger = logging.getLogger("flyrely.flight_reports")

RATE_LIMIT = 30
RATE_WINDOW_S = 3600
LOG_RETENTION = timedelta(days=7)
MAX_AGE = timedelta(days=30)

Answer = Literal["on_time", "late_15_44", "late_45_119", "late_120_plus", "cancelled", "did_not_fly"]

_DDL = """
CREATE TABLE IF NOT EXISTS flight_reports (
    user_id TEXT NOT NULL,
    flight_id TEXT NOT NULL,
    airline TEXT NOT NULL,
    flight_number TEXT NOT NULL,
    origin TEXT NOT NULL,
    dest TEXT NOT NULL,
    sched_dep_local TEXT NOT NULL,
    answer TEXT NOT NULL,
    tester INTEGER NOT NULL DEFAULT 0,
    created_utc TEXT NOT NULL,
    updated_utc TEXT NOT NULL,
    PRIMARY KEY (user_id, flight_id)
);
CREATE TABLE IF NOT EXISTS flight_report_log (
    ts_utc TEXT NOT NULL,
    user_id TEXT NOT NULL,
    flight_id TEXT NOT NULL,
    action TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_flight_report_log_user_ts ON flight_report_log (user_id, ts_utc);
"""

_lock = threading.Lock()
_ready_paths: set[str] = set()


def _now() -> datetime:
    """The clock; tests monkeypatch this."""
    return datetime.now(timezone.utc)


def db_path() -> Path:
    explicit = os.getenv("FLYRELY_FLIGHT_REPORTS_DB")
    if explicit:
        return Path(explicit)
    pred = Path(os.getenv("FLYRELY_PREDICTION_DB", "./data/predictions.sqlite"))
    return pred.parent / "flight_reports.sqlite"


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


class FlightReportIn(BaseModel):
    flight_id: str = Field(min_length=1, max_length=64)
    airline: str = Field(min_length=2, max_length=3)
    flight_number: str = Field(min_length=1, max_length=10)
    origin: str = Field(min_length=3, max_length=4)
    dest: str = Field(min_length=3, max_length=4)
    sched_dep_local: datetime  # ISO, naive local time at the origin airport
    answer: Optional[Answer] = None


async def require_user(authorization: Optional[str] = Header(default=None)) -> Tuple[str, str]:
    """(user id, email) of the signed-in Supabase user, or 401 / 503."""
    token = admin_auth._bearer(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail="Sign in required", headers={"WWW-Authenticate": "Bearer"})
    user_id, email = await admin_auth._lookup_user(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id, email


def _is_tester(email: str) -> bool:
    e = email.strip().lower()
    return e in admin_auth.admin_emails() or e.endswith("@example.com")


def _check_departure(dep_local: datetime, origin: str, now: datetime) -> None:
    """422 if departure is in the future or more than 30 days ago. Unknown airport zone: date-only 30-day check."""
    dep_utc = prediction_log.to_utc(dep_local, origin)
    if dep_utc is not None:
        if dep_utc > now:
            raise HTTPException(status_code=422, detail="That flight has not departed yet")
        if now - dep_utc > MAX_AGE:
            raise HTTPException(status_code=422, detail="That flight was more than 30 days ago")
    elif dep_local.date() < (now - MAX_AGE).date():
        raise HTTPException(status_code=422, detail="That flight was more than 30 days ago")


def _rate_limited(conn: sqlite3.Connection, user_id: str, now: datetime) -> bool:
    cutoff = (now - timedelta(seconds=RATE_WINDOW_S)).isoformat()
    n = conn.execute("SELECT COUNT(*) FROM flight_report_log WHERE user_id = ? AND ts_utc > ?",
                     (user_id, cutoff)).fetchone()[0]
    return n >= RATE_LIMIT


async def post_flight_report(body: FlightReportIn, user: Tuple[str, str] = Depends(require_user)):
    user_id, email = user
    if body.sched_dep_local.tzinfo is not None:
        raise HTTPException(status_code=422, detail="sched_dep_local must be a local time without a UTC offset")
    origin, dest = body.origin.strip().upper(), body.dest.strip().upper()
    now = _now()
    if body.answer is not None:
        _check_departure(body.sched_dep_local, origin, now)
    ts = now.isoformat()
    with _lock, closing(_connect()) as conn:
        if _rate_limited(conn, user_id, now):
            raise HTTPException(status_code=429, detail="Too many answers; try again later",
                                headers={"Retry-After": str(RATE_WINDOW_S)})
        with conn:
            if body.answer is None:
                conn.execute("DELETE FROM flight_reports WHERE user_id = ? AND flight_id = ?",
                             (user_id, body.flight_id))
                action, status = "withdraw", "withdrawn"
            else:
                conn.execute(
                    "INSERT INTO flight_reports (user_id, flight_id, airline, flight_number, origin, dest, "
                    "sched_dep_local, answer, tester, created_utc, updated_utc) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(user_id, flight_id) DO UPDATE SET airline=excluded.airline, "
                    "flight_number=excluded.flight_number, origin=excluded.origin, dest=excluded.dest, "
                    "sched_dep_local=excluded.sched_dep_local, answer=excluded.answer, tester=excluded.tester, "
                    "updated_utc=excluded.updated_utc",
                    (user_id, body.flight_id, body.airline.strip().upper(), body.flight_number.strip(), origin, dest,
                     body.sched_dep_local.isoformat(), body.answer, int(_is_tester(email)), ts, ts))
                action, status = "save", "saved"
            conn.execute("DELETE FROM flight_report_log WHERE ts_utc < ?", ((now - LOG_RETENTION).isoformat(),))
            conn.execute("INSERT INTO flight_report_log (ts_utc, user_id, flight_id, action) VALUES (?, ?, ?, ?)",
                         (ts, user_id, body.flight_id, action))
    return {"status": status}


async def list_flight_reports(limit: int = Query(default=100, ge=1, le=500),
                              _admin: str = Depends(admin_auth.require_admin)):
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT * FROM flight_reports ORDER BY updated_utc DESC LIMIT ?", (limit,)).fetchall()
    items = [dict(r) for r in rows]
    return {"count": len(items), "items": items}


def mount(app: FastAPI) -> None:
    """Register the routes directly on `app` (not via include_router, so app.routes stays a flat list of paths)."""
    app.add_api_route("/flight-reports", post_flight_report, methods=["POST"])
    app.add_api_route("/flight-reports", list_flight_reports, methods=["GET"])
