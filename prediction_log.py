"""Durable prediction log.

Every successful /predict call is written as one row so predictions can later
be matched against real outcomes. SQLite by default (FLYRELY_PREDICTION_DB);
Postgres when DATABASE_URL is set (see migrations/001_predictions.sql).
Writes are best-effort: callers run them in a background task and failures are
logged, never raised.
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("flyrely.prediction_log")

# IANA zones for the airports FlyRely has coordinates for (main.AIRPORT_COORDS).
# Airports not listed get sched_dep_utc = NULL rather than a guess.
AIRPORT_TZ = {
    "ATL": "America/New_York", "ORD": "America/Chicago", "DFW": "America/Chicago",
    "DEN": "America/Denver", "LAX": "America/Los_Angeles", "JFK": "America/New_York",
    "SFO": "America/Los_Angeles", "SEA": "America/Los_Angeles", "MIA": "America/New_York",
    "PHX": "America/Phoenix", "LAS": "America/Los_Angeles", "MCO": "America/New_York",
    "CLT": "America/New_York", "EWR": "America/New_York", "BOS": "America/New_York",
    "MSP": "America/Chicago", "DTW": "America/New_York", "PHL": "America/New_York",
    "LGA": "America/New_York", "DCA": "America/New_York", "IAH": "America/Chicago",
    "SLC": "America/Denver", "SAN": "America/Los_Angeles", "TPA": "America/New_York",
    "PDX": "America/Los_Angeles", "BWI": "America/New_York", "FLL": "America/New_York",
    "MDW": "America/Chicago", "BNA": "America/Chicago", "AUS": "America/Chicago",
}

COLUMNS = [
    "prediction_id", "ts_utc", "flight_id", "client", "origin", "dest",
    "sched_dep_local", "sched_dep_utc", "marketing_airline", "marketing_number",
    "operating_airline", "operating_number", "horizon", "minutes_to_departure",
    "delay_probability", "risk_level", "model_version", "rates_as_of",
    "features_json", "live_inputs_json", "weather_available",
]

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    ts_utc TEXT NOT NULL,
    flight_id TEXT NOT NULL,
    client TEXT NOT NULL,
    origin TEXT NOT NULL,
    dest TEXT NOT NULL,
    sched_dep_local TEXT NOT NULL,
    sched_dep_utc TEXT,
    marketing_airline TEXT,
    marketing_number TEXT,
    operating_airline TEXT,
    operating_number TEXT,
    horizon TEXT NOT NULL,
    minutes_to_departure INTEGER,
    delay_probability REAL NOT NULL,
    risk_level TEXT NOT NULL,
    model_version TEXT NOT NULL,
    rates_as_of TEXT,
    features_json TEXT NOT NULL,
    live_inputs_json TEXT,
    weather_available INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_flight_id ON predictions (flight_id);
CREATE INDEX IF NOT EXISTS idx_predictions_sched_dep_utc ON predictions (sched_dep_utc);
"""

_PG_DDL = [
    """CREATE TABLE IF NOT EXISTS predictions (
        prediction_id uuid PRIMARY KEY,
        ts_utc timestamptz NOT NULL,
        flight_id text NOT NULL,
        client text NOT NULL,
        origin text NOT NULL,
        dest text NOT NULL,
        sched_dep_local timestamp NOT NULL,
        sched_dep_utc timestamptz,
        marketing_airline text,
        marketing_number text,
        operating_airline text,
        operating_number text,
        horizon text NOT NULL,
        minutes_to_departure integer,
        delay_probability double precision NOT NULL,
        risk_level text NOT NULL,
        model_version text NOT NULL,
        rates_as_of date,
        features_json jsonb NOT NULL,
        live_inputs_json jsonb,
        weather_available boolean NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_predictions_flight_id ON predictions (flight_id)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_sched_dep_utc ON predictions (sched_dep_utc)",
    # Server-side only: RLS on with no policies, and no PostgREST role access.
    "ALTER TABLE predictions ENABLE ROW LEVEL SECURITY",
    """DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
            REVOKE ALL ON predictions FROM anon;
        END IF;
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
            REVOKE ALL ON predictions FROM authenticated;
        END IF;
    END $$""",
]

# Postgres safety: bound every connection/statement, and stop writing after
# repeated failures so a dead DB cannot pin threadpool threads.
PG_CONNECT_KWARGS = {"connect_timeout": 5, "options": "-c statement_timeout=5000"}
BREAKER_THRESHOLD = 3
BREAKER_COOLDOWN_S = 60


def to_utc(local_dt: datetime, airport: str) -> Optional[datetime]:
    """Convert a naive local departure time to UTC using the origin airport's zone, or None if unknown."""
    tz = AIRPORT_TZ.get(airport)
    if not tz:
        return None
    # DST: zoneinfo uses fold=0, so an ambiguous fall-back time resolves to the first
    # (daylight) occurrence, and a non-existent spring-forward time is read with the
    # pre-transition offset. The result can be off by up to an hour on those two nights.
    return local_dt.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)


def build_record(*, flight_id: str, origin: str, dest: str, sched_dep_local: datetime,
                 marketing_airline: Optional[str], marketing_number: Optional[str],
                 operating_airline: Optional[str], operating_number: Optional[str],
                 delay_probability: float, risk_level: str, model_version: str,
                 features: dict, weather_available: bool, horizon: str = "booking",
                 rates_as_of: Optional[date] = None, live_inputs: Optional[dict] = None,
                 now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    dep_utc = to_utc(sched_dep_local, origin)
    minutes = int((dep_utc - now).total_seconds() // 60) if dep_utc else None
    return {
        "prediction_id": str(uuid.uuid4()),
        "ts_utc": now,
        "flight_id": flight_id,
        "client": "h2h" if flight_id.startswith("h2h-") else "app",
        "origin": origin,
        "dest": dest,
        "sched_dep_local": sched_dep_local,
        "sched_dep_utc": dep_utc,
        "marketing_airline": marketing_airline,
        "marketing_number": marketing_number,
        "operating_airline": operating_airline,
        "operating_number": operating_number,
        "horizon": horizon,
        "minutes_to_departure": minutes,
        "delay_probability": delay_probability,
        "risk_level": risk_level,
        "model_version": model_version,
        "rates_as_of": rates_as_of,
        "features_json": features,
        "live_inputs_json": live_inputs,
        "weather_available": weather_available,
    }


def _dumps(obj) -> Optional[str]:
    if obj is None:
        return None
    return json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o))


class PredictionStore:
    """SQLite (default) or Postgres (when DATABASE_URL is set) prediction log."""

    def __init__(self, sqlite_path: Optional[str] = None, database_url: Optional[str] = None):
        self.database_url = database_url if database_url is not None else os.getenv("DATABASE_URL")
        self.sqlite_path = Path(sqlite_path or os.getenv("FLYRELY_PREDICTION_DB", "./data/predictions.sqlite"))
        self._ready = False
        self._failures = 0
        self._paused_until = 0.0

    def _connect_pg(self):
        import psycopg
        return psycopg.connect(self.database_url, **PG_CONNECT_KWARGS)

    @property
    def backend(self) -> str:
        return "postgres" if self.database_url else "sqlite"

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        if self.database_url:
            with self._connect_pg() as conn:
                for stmt in _PG_DDL:
                    conn.execute(stmt)
        else:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect_sqlite()) as conn:
                conn.executescript(_SQLITE_DDL)
        self._ready = True

    def _connect_sqlite(self):
        conn = sqlite3.connect(self.sqlite_path, timeout=5)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def write(self, rec: dict) -> None:
        """Insert one row. Raises on failure; use safe_write for best-effort."""
        self._ensure_schema()
        cols = ", ".join(COLUMNS)
        row = dict(rec)
        if self.database_url:
            from psycopg.types.json import Jsonb
            for k in ("features_json", "live_inputs_json"):
                if row[k] is not None:
                    row[k] = Jsonb(json.loads(_dumps(row[k])))
            with self._connect_pg() as conn:
                conn.execute(
                    f"INSERT INTO predictions ({cols}) VALUES ({', '.join(['%s'] * len(COLUMNS))})",
                    [row[c] for c in COLUMNS])
        else:
            for k in ("ts_utc", "sched_dep_local", "sched_dep_utc", "rates_as_of"):
                if row[k] is not None:
                    row[k] = row[k].isoformat()
            row["features_json"] = _dumps(row["features_json"])
            row["live_inputs_json"] = _dumps(row["live_inputs_json"])
            row["weather_available"] = int(row["weather_available"])
            with closing(self._connect_sqlite()) as conn:
                with conn:
                    conn.execute(
                        f"INSERT INTO predictions ({cols}) VALUES ({', '.join(['?'] * len(COLUMNS))})",
                        [row[c] for c in COLUMNS])

    def safe_write(self, rec: dict) -> None:
        """Best-effort write with a circuit breaker: after BREAKER_THRESHOLD consecutive
        failures, skip writes for BREAKER_COOLDOWN_S seconds."""
        if time.monotonic() < self._paused_until:
            return
        try:
            self.write(rec)
            self._failures = 0
        except Exception as e:
            self._failures += 1
            logger.warning(f"Prediction log write failed: {type(e).__name__}: {e}")
            if self._failures >= BREAKER_THRESHOLD:
                self._paused_until = time.monotonic() + BREAKER_COOLDOWN_S
                self._failures = 0
                logger.warning(f"Prediction log paused for {BREAKER_COOLDOWN_S}s after "
                               f"{BREAKER_THRESHOLD} consecutive failures")
