"""Push watch storage (plan §4.4, §8). SQLite by default (FLYRELY_PREDICTION_DB, same file as the prediction log),
Postgres when DATABASE_URL is set (see migrations/003_copilot_push.sql), like prediction_log.PredictionStore.

Stores only what the heads-up needs: the push endpoint and keys, the flight's public schedule identifiers,
`client_flight_id`, the last risk the watcher saw, and what was already sent. Never the home airport, an account
id, an email or an IP. Endpoints and keys are never logged. Rows are deleted at `expires_at`, on 404/410, or on
unsubscribe.
"""
import logging
import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from prediction_log import PG_CONNECT_KWARGS, PredictionStore

logger = logging.getLogger("flyrely.copilot.push_store")

MAX_WATCHES_PER_ENDPOINT = 20
# The departure is a local wall-clock time and the airport's zone is not stored, so the true UTC instant is
# within [local - 14h, local + 12h]. A watch lives until local-as-UTC + 24h + 12h: past the departure everywhere.
EXPIRY_AFTER_DEPARTURE = timedelta(hours=36)
ALLOWED_HOST_SUFFIXES = (".push.apple.com", "fcm.googleapis.com", "updates.push.services.mozilla.com",
                         ".push.services.mozilla.com", ".notify.windows.com")
KIND_HEADS_UP = "at_risk_heads_up"
KIND_CANCELLED = "cancelled"
_SENT_COLUMN = {KIND_HEADS_UP: "heads_up_sent_at", KIND_CANCELLED: "cancelled_sent_at"}

COLUMNS = ["watch_id", "endpoint", "p256dh", "auth", "client_flight_id", "airline", "flight_number", "origin",
           "destination", "departure_time", "expires_at", "risk_level", "delay_probability", "created_at",
           "heads_up_sent_at", "cancelled_sent_at"]

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS copilot_push_watches (
    watch_id TEXT PRIMARY KEY,
    endpoint TEXT NOT NULL,
    p256dh TEXT NOT NULL,
    auth TEXT NOT NULL,
    client_flight_id TEXT,
    airline TEXT NOT NULL,
    flight_number TEXT NOT NULL,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    departure_time TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    risk_level TEXT,
    delay_probability REAL,
    created_at TEXT NOT NULL,
    heads_up_sent_at TEXT,
    cancelled_sent_at TEXT,
    UNIQUE (endpoint, airline, flight_number, origin, departure_time)
);
CREATE INDEX IF NOT EXISTS idx_copilot_push_expires ON copilot_push_watches (expires_at);
CREATE INDEX IF NOT EXISTS idx_copilot_push_endpoint ON copilot_push_watches (endpoint);
"""

_PG_DDL = [
    """CREATE TABLE IF NOT EXISTS copilot_push_watches (
        watch_id text PRIMARY KEY,
        endpoint text NOT NULL,
        p256dh text NOT NULL,
        auth text NOT NULL,
        client_flight_id text,
        airline text NOT NULL,
        flight_number text NOT NULL,
        origin text NOT NULL,
        destination text NOT NULL,
        departure_time timestamp NOT NULL,
        expires_at timestamptz NOT NULL,
        risk_level text,
        delay_probability double precision,
        created_at timestamptz NOT NULL,
        heads_up_sent_at timestamptz,
        cancelled_sent_at timestamptz,
        UNIQUE (endpoint, airline, flight_number, origin, departure_time)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_copilot_push_expires ON copilot_push_watches (expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_copilot_push_endpoint ON copilot_push_watches (endpoint)",
    "ALTER TABLE copilot_push_watches ENABLE ROW LEVEL SECURITY",
    """DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
            REVOKE ALL ON copilot_push_watches FROM anon;
        END IF;
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
            REVOKE ALL ON copilot_push_watches FROM authenticated;
        END IF;
    END $$""",
]


class TooManyWatchesTotal(Exception):
    """The global cap on active watches is reached."""


class TooManyWatches(Exception):
    """The endpoint already has MAX_WATCHES_PER_ENDPOINT active watches."""


def endpoint_allowed(endpoint: str) -> bool:
    """SSRF guard: https only, and the host must be a known push service."""
    from urllib.parse import urlsplit
    try:
        parts = urlsplit(endpoint)
        host = (parts.hostname or "").lower()
        if parts.scheme != "https" or not host or parts.username or parts.password or parts.port not in (None, 443):
            return False
    except ValueError:
        return False
    return any(host == s.lstrip(".") or (s.startswith(".") and host.endswith(s)) for s in ALLOWED_HOST_SUFFIXES)


def departure_dt(value: str) -> datetime:
    """Local wall-clock ISO -> naive datetime (any offset is dropped, as /predict does)."""
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return dt.replace(tzinfo=None)


def expiry_for(departure: datetime) -> datetime:
    return departure.replace(tzinfo=timezone.utc) + EXPIRY_AFTER_DEPARTURE


@dataclass(frozen=True)
class Watch:
    watch_id: str
    endpoint: str
    p256dh: str
    auth: str
    client_flight_id: Optional[str]
    airline: str
    flight_number: str
    origin: str
    destination: str
    departure_time: datetime          # naive local
    expires_at: datetime              # aware UTC
    risk_level: Optional[str]
    delay_probability: Optional[float]
    heads_up_sent_at: Optional[datetime]
    cancelled_sent_at: Optional[datetime]

    def __repr__(self) -> str:        # never leak the endpoint or keys into logs
        return f"Watch({self.watch_id} {self.flight_number} {self.origin}-{self.destination})"

    @property
    def subscription(self) -> dict:
        return {"endpoint": self.endpoint, "keys": {"p256dh": self.p256dh, "auth": self.auth}}


def _utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _naive(value) -> datetime:
    return datetime.fromisoformat(value) if isinstance(value, str) else value.replace(tzinfo=None)


class PushStore:
    def __init__(self, sqlite_path: Optional[str] = None, database_url: Optional[str] = None):
        # Reuse PredictionStore for backend selection and the SQLite path.
        self._base = PredictionStore(sqlite_path=sqlite_path, database_url=database_url)
        self._ready = False

    @property
    def backend(self) -> str:
        return self._base.backend

    # ------------------------------------------------------------------ plumbing
    def _pg(self) -> bool:
        return bool(self._base.database_url)

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        if self._pg():
            import psycopg
            with psycopg.connect(self._base.database_url, **PG_CONNECT_KWARGS) as conn:
                for stmt in _PG_DDL:
                    conn.execute(stmt)
        else:
            self._base.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._base._connect_sqlite()) as conn:
                conn.executescript(_SQLITE_DDL)
        self._ready = True

    def _exec(self, sql: str, params: Iterable = (), fetch: bool = False):
        """Run one statement in its own transaction. `sql` uses `?`; returns (rows, rowcount)."""
        self._ensure_schema()
        params = list(params)
        if self._pg():
            import psycopg
            with psycopg.connect(self._base.database_url, **PG_CONNECT_KWARGS) as conn:
                cur = conn.execute(sql.replace("?", "%s"), params)
                return (cur.fetchall() if fetch else []), cur.rowcount
        with closing(self._base._connect_sqlite()) as conn:
            with conn:
                cur = conn.execute(sql, params)
                return (cur.fetchall() if fetch else []), cur.rowcount

    def _ts(self, dt: Optional[datetime]):
        """Bind a datetime: native on Postgres, ISO text (UTC, fixed width) on SQLite."""
        if dt is None or self._pg():
            return dt
        return _utc(dt).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    def _local(self, dt: datetime):
        return dt if self._pg() else dt.replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S")

    def _row(self, r) -> Watch:
        d = dict(zip(COLUMNS, r))
        return Watch(
            watch_id=d["watch_id"], endpoint=d["endpoint"], p256dh=d["p256dh"], auth=d["auth"],
            client_flight_id=d["client_flight_id"], airline=d["airline"], flight_number=d["flight_number"],
            origin=d["origin"], destination=d["destination"], departure_time=_naive(d["departure_time"]),
            expires_at=_utc(d["expires_at"]), risk_level=d["risk_level"], delay_probability=d["delay_probability"],
            heads_up_sent_at=_utc(d["heads_up_sent_at"]), cancelled_sent_at=_utc(d["cancelled_sent_at"]),
        )

    # ------------------------------------------------------------------ API
    def subscribe(self, *, endpoint: str, p256dh: str, auth: str, airline: str, flight_number: str, origin: str,
                  destination: str, departure: datetime, client_flight_id: Optional[str] = None,
                  now: Optional[datetime] = None, max_total: Optional[int] = None) -> Watch:
        """Idempotent on (endpoint, flight): a repeat call refreshes the keys and returns the same watch."""
        now = now or datetime.now(timezone.utc)
        key = (endpoint, airline, flight_number, origin, self._local(departure))
        where = "endpoint = ? AND airline = ? AND flight_number = ? AND origin = ? AND departure_time = ?"
        select = f"SELECT {', '.join(COLUMNS)} FROM copilot_push_watches WHERE "
        rows, _ = self._exec(select + where, key, fetch=True)
        if rows:
            self._exec("UPDATE copilot_push_watches SET p256dh = ?, auth = ?, "
                       "client_flight_id = COALESCE(?, client_flight_id) WHERE " + where,
                       (p256dh, auth, client_flight_id, *key))
        else:
            count, _ = self._exec("SELECT COUNT(*) FROM copilot_push_watches WHERE endpoint = ? AND expires_at > ?",
                                  (endpoint, self._ts(now)), fetch=True)
            if count[0][0] >= MAX_WATCHES_PER_ENDPOINT:
                raise TooManyWatches()
            if max_total is not None:
                total, _ = self._exec("SELECT COUNT(*) FROM copilot_push_watches WHERE expires_at > ?",
                                      (self._ts(now),), fetch=True)
                if total[0][0] >= max_total:
                    raise TooManyWatchesTotal()
            self._exec(
                f"INSERT INTO copilot_push_watches ({', '.join(COLUMNS)}) "
                f"VALUES ({', '.join(['?'] * len(COLUMNS))}) ON CONFLICT DO NOTHING",
                ("w_" + secrets.token_hex(8), endpoint, p256dh, auth, client_flight_id, airline, flight_number,
                 origin, destination, self._local(departure), self._ts(expiry_for(departure)), None, None,
                 self._ts(now), None, None))
        rows, _ = self._exec(select + where, key, fetch=True)
        return self._row(rows[0])

    def unsubscribe(self, endpoint: str, watch_id: Optional[str] = None) -> int:
        if watch_id:
            _, n = self._exec("DELETE FROM copilot_push_watches WHERE endpoint = ? AND watch_id = ?",
                              (endpoint, watch_id))
        else:
            _, n = self._exec("DELETE FROM copilot_push_watches WHERE endpoint = ?", (endpoint,))
        return n

    def delete_endpoint(self, endpoint: str) -> int:
        """The push service said the subscription is gone (404/410): drop every watch for it."""
        return self.unsubscribe(endpoint)

    def delete_expired(self, now: Optional[datetime] = None) -> int:
        """Retention: remove watches whose flight is in the past."""
        _, n = self._exec("DELETE FROM copilot_push_watches WHERE expires_at <= ?",
                          (self._ts(now or datetime.now(timezone.utc)),))
        return n

    def active(self, now: Optional[datetime] = None) -> list[Watch]:
        rows, _ = self._exec(f"SELECT {', '.join(COLUMNS)} FROM copilot_push_watches WHERE expires_at > ? "
                             "ORDER BY departure_time", (self._ts(now or datetime.now(timezone.utc)),), fetch=True)
        return [self._row(r) for r in rows]

    def record_risk(self, watch_id: str, level: str, probability: float) -> None:
        self._exec("UPDATE copilot_push_watches SET risk_level = ?, delay_probability = ? WHERE watch_id = ?",
                   (level, probability, watch_id))

    def claim(self, watch_id: str, kind: str, now: Optional[datetime] = None) -> bool:
        """Atomically mark `kind` as sent. True for exactly one caller, so two processes cannot both send."""
        col = _SENT_COLUMN[kind]
        _, n = self._exec(f"UPDATE copilot_push_watches SET {col} = ? WHERE watch_id = ? AND {col} IS NULL",
                          (self._ts(now or datetime.now(timezone.utc)), watch_id))
        return n == 1

    def release(self, watch_id: str, kind: str) -> None:
        """Undo a claim after a transient send failure so the next cycle retries."""
        col = _SENT_COLUMN[kind]
        self._exec(f"UPDATE copilot_push_watches SET {col} = NULL WHERE watch_id = ?", (watch_id,))

    def count(self) -> int:
        rows, _ = self._exec("SELECT COUNT(*) FROM copilot_push_watches", fetch=True)
        return rows[0][0]
