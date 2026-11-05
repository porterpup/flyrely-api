"""Final flight outcomes for live accuracy tracking.

`outcomes` holds what actually happened to a flight (BTS on-time records are the final,
training-label source). SQLite by default, Postgres when DATABASE_URL is set, exactly like
prediction_log.PredictionStore (see migrations/002_outcomes.sql).

Matching a logged prediction to an outcome: operating carrier + flight number + origin + local
scheduled departure date, falling back to marketing carrier + number when the operator is unknown.
"""

import logging
import re
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from prediction_log import PredictionStore

logger = logging.getLogger("flyrely.outcomes")

COLUMNS = [
    "carrier", "flight_number", "origin", "sched_dep_date", "source", "marketing_airline",
    "marketing_number", "dest", "sched_dep_local", "dep_delay_min", "cancelled", "diverted",
    "label_d15", "label_quality", "fetched_utc",
]
KEY = ["carrier", "flight_number", "origin", "sched_dep_date", "source"]
SOURCE_PRIORITY = {"bts": 0, "aeroapi": 1, "swim": 2}  # final first, provisional after

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS outcomes (
    carrier TEXT NOT NULL,
    flight_number TEXT NOT NULL,
    origin TEXT NOT NULL,
    sched_dep_date TEXT NOT NULL,
    source TEXT NOT NULL,
    marketing_airline TEXT,
    marketing_number TEXT,
    dest TEXT,
    sched_dep_local TEXT,
    dep_delay_min REAL,
    cancelled INTEGER NOT NULL,
    diverted INTEGER NOT NULL,
    label_d15 INTEGER,
    label_quality TEXT NOT NULL,
    fetched_utc TEXT NOT NULL,
    PRIMARY KEY (carrier, flight_number, origin, sched_dep_date, source)
);
CREATE INDEX IF NOT EXISTS idx_outcomes_marketing
    ON outcomes (marketing_airline, marketing_number, origin, sched_dep_date);
"""

_PG_DDL = [
    """CREATE TABLE IF NOT EXISTS outcomes (
        carrier text NOT NULL,
        flight_number text NOT NULL,
        origin text NOT NULL,
        sched_dep_date date NOT NULL,
        source text NOT NULL,
        marketing_airline text,
        marketing_number text,
        dest text,
        sched_dep_local timestamp,
        dep_delay_min double precision,
        cancelled boolean NOT NULL,
        diverted boolean NOT NULL,
        label_d15 smallint,
        label_quality text NOT NULL,
        fetched_utc timestamptz NOT NULL,
        PRIMARY KEY (carrier, flight_number, origin, sched_dep_date, source)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_marketing ON outcomes "
    "(marketing_airline, marketing_number, origin, sched_dep_date)",
    "ALTER TABLE outcomes ENABLE ROW LEVEL SECURITY",
    """DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
            REVOKE ALL ON outcomes FROM anon;
        END IF;
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
            REVOKE ALL ON outcomes FROM authenticated;
        END IF;
    END $$""",
]

PRED_COLUMNS = [
    "prediction_id", "ts_utc", "flight_id", "client", "origin", "dest", "sched_dep_local",
    "marketing_airline", "marketing_number", "operating_airline", "operating_number",
    "horizon", "minutes_to_departure", "delay_probability", "model_version",
]


def norm_number(value) -> Optional[str]:
    """'DL1234', ' 01234', 1234, 1234.0 -> '1234'; None if there are no digits."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return None
    s = str(value)
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    m = re.search(r"(\d+)", s)
    return str(int(m.group(1))) if m else None


class OutcomeStore:
    """Outcomes table next to the prediction log; shares its backend and connection settings."""

    def __init__(self, store: Optional[PredictionStore] = None):
        self.store = store or PredictionStore()
        self._ready = False

    @property
    def backend(self) -> str:
        return self.store.backend

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        if self.store.database_url:
            with self.store._connect_pg() as conn:
                for stmt in _PG_DDL:
                    conn.execute(stmt)
        else:
            self.store.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self.store._connect_sqlite()) as conn:
                conn.executescript(_SQLITE_DDL)
        self._ready = True

    def upsert(self, df: pd.DataFrame) -> int:
        """Insert or replace outcome rows (same key and source). Returns the row count."""
        if df.empty:
            return 0
        self._ensure_schema()
        rows = []
        for r in df[COLUMNS].to_dict("records"):
            r = {k: (None if pd.isna(v) else v) for k, v in r.items()}
            for k in ("cancelled", "diverted"):
                r[k] = bool(r[k])
            if r["label_d15"] is not None:
                r["label_d15"] = int(r["label_d15"])
            rows.append(r)
        cols = ", ".join(COLUMNS)
        updates = ", ".join(f"{c} = excluded.{c}" for c in COLUMNS if c not in KEY)
        conflict = f"ON CONFLICT ({', '.join(KEY)}) DO UPDATE SET {updates}"
        if self.store.database_url:
            for r in rows:
                r["sched_dep_date"] = pd.Timestamp(r["sched_dep_date"]).date()
                r["sched_dep_local"] = (pd.Timestamp(r["sched_dep_local"]).to_pydatetime()
                                        if r["sched_dep_local"] is not None else None)
                r["fetched_utc"] = pd.Timestamp(r["fetched_utc"]).to_pydatetime()
            with self.store._connect_pg() as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        f"INSERT INTO outcomes ({cols}) VALUES ({', '.join(['%s'] * len(COLUMNS))}) {conflict}",
                        [[r[c] for c in COLUMNS] for r in rows])
        else:
            for r in rows:
                r["sched_dep_date"] = pd.Timestamp(r["sched_dep_date"]).date().isoformat()
                if r["sched_dep_local"] is not None:
                    r["sched_dep_local"] = pd.Timestamp(r["sched_dep_local"]).isoformat()
                r["fetched_utc"] = pd.Timestamp(r["fetched_utc"]).isoformat()
                r["cancelled"] = int(r["cancelled"])
                r["diverted"] = int(r["diverted"])
            with closing(self.store._connect_sqlite()) as conn:
                with conn:
                    conn.executemany(
                        f"INSERT INTO outcomes ({cols}) VALUES ({', '.join(['?'] * len(COLUMNS))}) {conflict}",
                        [[r[c] for c in COLUMNS] for r in rows])
        return len(rows)

    def read(self) -> pd.DataFrame:
        self._ensure_schema()
        return _query_df(self.store, f"SELECT {', '.join(COLUMNS)} FROM outcomes", COLUMNS)


def _query_df(store: PredictionStore, sql: str, columns: list) -> pd.DataFrame:
    if store.database_url:
        with store._connect_pg() as conn:
            rows = conn.execute(sql).fetchall()
    else:
        with closing(store._connect_sqlite()) as conn:
            rows = conn.execute(sql).fetchall()
    return pd.DataFrame(rows, columns=columns)


def read_predictions(store: PredictionStore, since: Optional[str] = None) -> pd.DataFrame:
    """Logged predictions as a DataFrame, with prepared matching columns. Empty if the table doesn't exist yet."""
    try:
        store._ensure_schema()
        df = _query_df(store, f"SELECT {', '.join(PRED_COLUMNS)} FROM predictions", PRED_COLUMNS)
    except Exception as e:  # never created, or DB unreachable: let callers decide
        logger.warning(f"Could not read predictions: {type(e).__name__}: {e}")
        raise
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
    local = pd.to_datetime(df["sched_dep_local"], format="ISO8601")
    df["sched_dep_local"] = local.dt.tz_localize(None) if local.dt.tz is not None else local
    df["delay_probability"] = df["delay_probability"].astype(float)
    if since:
        df = df[df["sched_dep_local"] >= pd.Timestamp(since)].reset_index(drop=True)
    return df


def _prep_pred_keys(preds: pd.DataFrame) -> pd.DataFrame:
    p = preds.copy()
    p["_date"] = p["sched_dep_local"].dt.normalize()
    p["_op_c"] = p["operating_airline"]
    p["_op_n"] = p["operating_number"].map(norm_number)
    p["_mk_c"] = p["marketing_airline"]
    p["_mk_n"] = p["marketing_number"].map(norm_number)
    return p


def match_predictions(preds: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Left-join outcomes onto predictions; adds `matched`, `label_d15`, `cancelled`, `diverted`,
    `dep_delay_min`, `outcome_source`. Operating key first, then marketing key. Final sources win over provisional."""
    p = _prep_pred_keys(preds)
    out_cols = ["label_d15", "cancelled", "diverted", "dep_delay_min", "source"]
    o = outcomes.copy()
    o["_date"] = pd.to_datetime(o["sched_dep_date"]).dt.normalize()
    o["_prio"] = o["source"].map(SOURCE_PRIORITY).fillna(99)
    o = o.sort_values("_prio")
    o["flight_number"] = o["flight_number"].map(norm_number)
    o["marketing_number"] = o["marketing_number"].map(norm_number)

    p["matched"] = False
    for c in out_cols:
        p[c] = pd.NA
    p = p.reset_index(drop=True)
    p["_row"] = p.index

    passes = [("_op_c", "_op_n", "carrier", "flight_number"),
              ("_mk_c", "_mk_n", "marketing_airline", "marketing_number")]
    for pc, pn, oc, on in passes:
        todo = p[~p["matched"] & p[pc].notna() & p[pn].notna()]
        if todo.empty:
            continue
        cand = o[o[oc].notna() & o[on].notna()].drop_duplicates([oc, on, "origin", "_date"])
        m = todo[["_row", pc, pn, "origin", "_date"]].merge(
            cand[[oc, on, "origin", "_date"] + out_cols],
            left_on=[pc, pn, "origin", "_date"], right_on=[oc, on, "origin", "_date"], how="inner")
        if m.empty:
            continue
        idx = m["_row"].to_numpy()
        for c in out_cols:
            p.loc[idx, c] = m[c].to_numpy()
        p.loc[idx, "matched"] = True
    p = p.rename(columns={"source": "outcome_source"})
    return p.drop(columns=[c for c in p.columns if c.startswith("_")])


def load_bts_outcomes(preds: pd.DataFrame, bts_dir: Path, now: Optional[datetime] = None) -> pd.DataFrame:
    """Outcome rows from local BTS per-month Parquet (backtest/prepare.py output) for flights we predicted.

    Reads only months containing predicted dates, so it needs no network. Keeps only BTS rows that match
    some prediction, so the outcomes table stays small. dep delay >= 15 is the training label."""
    now = now or datetime.now(timezone.utc)
    bts_dir = Path(bts_dir)
    p = _prep_pred_keys(preds)
    months = sorted({f"{d:%Y-%m}" for d in p["_date"].dropna().unique()})
    frames = []
    for mth in months:
        f = bts_dir / f"{mth}.parquet"
        if not f.exists():
            logger.info(f"No BTS file for {mth} ({f}); predictions in that month stay unmatched")
            continue
        df = pd.read_parquet(f)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    b = pd.concat(frames, ignore_index=True)
    hhmm = b["CRSDepTime"].fillna(0).astype(int).clip(upper=2359)
    b["sched_dep_date"] = pd.to_datetime(b["FlightDate"]).dt.normalize()
    b["sched_dep_local"] = b["sched_dep_date"] + pd.to_timedelta(hhmm // 100 * 60 + hhmm % 100, unit="m")
    cancelled = b["Cancelled"].fillna(0).astype(bool)
    diverted = b["Diverted"].fillna(0).astype(bool)
    delay = b["DepDelay"].astype(float)
    mk_num = b["Flight_Number_Marketing_Airline"] if "Flight_Number_Marketing_Airline" in b else None
    rows = pd.DataFrame({
        "carrier": b["Reporting_Airline"].astype(str),
        "flight_number": b["Flight_Number_Reporting_Airline"].map(norm_number),
        "origin": b["Origin"].astype(str),
        "sched_dep_date": b["sched_dep_date"],
        "source": "bts",
        "marketing_airline": b["Marketing_Airline_Network"].astype(str) if "Marketing_Airline_Network" in b else None,
        "marketing_number": mk_num.map(norm_number) if mk_num is not None else None,
        "dest": b["Dest"].astype(str),
        "sched_dep_local": b["sched_dep_local"],
        "dep_delay_min": delay,
        "cancelled": cancelled,
        "diverted": diverted,
        "label_d15": (delay >= 15).astype(int).where(~cancelled & delay.notna()),
        "label_quality": "exact",
        "fetched_utc": now,
    })
    rows = rows[rows["flight_number"].notna()]
    # keep only rows that some prediction matches (either key)
    keys_op = set(zip(p["_op_c"], p["_op_n"], p["origin"], p["_date"]))
    keys_mk = set(zip(p["_mk_c"], p["_mk_n"], p["origin"], p["_date"]))
    k_op = list(zip(rows["carrier"], rows["flight_number"], rows["origin"], rows["sched_dep_date"]))
    k_mk = list(zip(rows["marketing_airline"], rows["marketing_number"], rows["origin"], rows["sched_dep_date"]))
    keep = [a in keys_op or b_ in keys_mk for a, b_ in zip(k_op, k_mk)]
    rows = rows[keep].drop_duplicates(KEY)
    return rows.reset_index(drop=True)
