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
    # Single source for time zones: models/airports_ref.csv, loaded by main (imported lazily
    # because main imports this module). Airports not in the table get NULL rather than a guess.
    import main
    ref = main.AIRPORTS_REF.get(airport)
    tz = ref.get("tz") if ref else None
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
                 now: Optional[datetime] = None, arr_probability: Optional[float] = None,
                 cancel_probability: Optional[float] = None, arr_model_version: Optional[str] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    dep_utc = to_utc(sched_dep_local, origin)
    minutes = int((dep_utc - now).total_seconds() // 60) if dep_utc else None
    rec = {
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
    # Arrival columns (migration 004) are only written when there is an arrival forecast, so a database that has not
    # been migrated keeps working while the arrival flags are off.
    for k, v in (("arr_probability", arr_probability), ("cancel_probability", cancel_probability),
                 ("arr_model_version", arr_model_version)):
        if v is not None:
            rec[k] = v
    return rec


ARRIVAL_COLUMNS = {"arr_probability": ("REAL", "double precision"),
                   "cancel_probability": ("REAL", "double precision"),
                   "arr_model_version": ("TEXT", "text")}

ITIN_COLUMNS = ["itinerary_id", "ts_utc", "connection_index", "leg_flight_ids", "at", "kind", "layover_min",
                "mct_min", "mct_source", "miss_probability", "validated", "not_validated_reason", "model_version"]
_ITIN_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS itinerary_predictions (
    itinerary_id TEXT NOT NULL, ts_utc TEXT NOT NULL, connection_index INTEGER NOT NULL, leg_flight_ids TEXT NOT NULL,
    at TEXT NOT NULL, kind TEXT NOT NULL, layover_min INTEGER, mct_min INTEGER, mct_source TEXT,
    miss_probability REAL, validated INTEGER NOT NULL, not_validated_reason TEXT, model_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_itinerary_predictions_itinerary ON itinerary_predictions (itinerary_id);
"""
_ITIN_PG_DDL = [
    """CREATE TABLE IF NOT EXISTS itinerary_predictions (
        id bigint generated always as identity primary key,
        itinerary_id text NOT NULL, ts_utc timestamptz NOT NULL, connection_index integer NOT NULL,
        leg_flight_ids text[] NOT NULL, at text NOT NULL, kind text NOT NULL, layover_min integer, mct_min integer,
        mct_source text, miss_probability double precision, validated boolean NOT NULL,
        not_validated_reason text, model_version text NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_itinerary_predictions_itinerary ON itinerary_predictions (itinerary_id)",
    "ALTER TABLE itinerary_predictions ENABLE ROW LEVEL SECURITY",
]


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

    def _ensure_arrival_columns(self) -> None:
        """Add the migration-004 columns if missing (same self-migrating approach as the CREATE TABLE above)."""
        if getattr(self, "_arr_cols_ready", False):
            return
        if self.database_url:
            with self._connect_pg() as conn:
                for name, (_, pg) in ARRIVAL_COLUMNS.items():
                    conn.execute(f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {name} {pg}")
        else:
            with closing(self._connect_sqlite()) as conn:
                have = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
                with conn:
                    for name, (lite, _) in ARRIVAL_COLUMNS.items():
                        if name not in have:
                            conn.execute(f"ALTER TABLE predictions ADD COLUMN {name} {lite}")
        self._arr_cols_ready = True

    def write_itinerary(self, rows: list) -> None:
        """Insert one row per connection into itinerary_predictions. Raises on failure; use safe_write_itinerary."""
        if not rows:
            return
        if self.database_url:
            with self._connect_pg() as conn:
                if not getattr(self, "_itin_ready", False):
                    for stmt in _ITIN_PG_DDL:
                        conn.execute(stmt)
                    self._itin_ready = True
                for r in rows:
                    conn.execute(f"INSERT INTO itinerary_predictions ({', '.join(ITIN_COLUMNS)}) "
                                 f"VALUES ({', '.join(['%s'] * len(ITIN_COLUMNS))})", [r[c] for c in ITIN_COLUMNS])
        else:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect_sqlite()) as conn:
                conn.executescript(_ITIN_SQLITE_DDL)
                with conn:
                    for r in rows:
                        v = dict(r)
                        v["ts_utc"] = v["ts_utc"].isoformat()
                        v["leg_flight_ids"] = json.dumps(v["leg_flight_ids"])
                        v["validated"] = int(v["validated"])
                        conn.execute(f"INSERT INTO itinerary_predictions ({', '.join(ITIN_COLUMNS)}) "
                                     f"VALUES ({', '.join(['?'] * len(ITIN_COLUMNS))})", [v[c] for c in ITIN_COLUMNS])

    def safe_write_itinerary(self, rows: list) -> None:
        try:
            self.write_itinerary(rows)
        except Exception as e:
            logger.warning(f"Itinerary log write failed: {type(e).__name__}: {e}")

    def write(self, rec: dict) -> None:
        """Insert one row. Raises on failure; use safe_write for best-effort."""
        self._ensure_schema()
        extra = [c for c in ARRIVAL_COLUMNS if rec.get(c) is not None]
        if extra:
            self._ensure_arrival_columns()
        cols_list = COLUMNS + extra
        cols = ", ".join(cols_list)
        row = dict(rec)
        if self.database_url:
            from psycopg.types.json import Jsonb
            for k in ("features_json", "live_inputs_json"):
                if row[k] is not None:
                    row[k] = Jsonb(json.loads(_dumps(row[k])))
            with self._connect_pg() as conn:
                conn.execute(
                    f"INSERT INTO predictions ({cols}) VALUES ({', '.join(['%s'] * len(cols_list))})",
                    [row[c] for c in cols_list])
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
                        f"INSERT INTO predictions ({cols}) VALUES ({', '.join(['?'] * len(cols_list))})",
                        [row[c] for c in cols_list])

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
