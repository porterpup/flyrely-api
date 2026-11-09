"""
Month-by-month arrival data and caches (protocol: backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md).

    python backtest/arrival_data.py [--months 2025-01 2026-08] [--force]

Reads data/bts/parquet_arr one month at a time through backtest.load_months, computes the 18 committed features
(backtest.committed_features) and writes a float32 cache per month to data/cache/arrival/YYYY-MM.parquet: features,
labels, keys, and the hub-local wall-clock scheduled departure/arrival used by the proxy-pair builder (pairs.py).
Months before 2025-01 and from 2026-09 on are refused; the script exits if FLYRELY_HOLDOUT_UNLOCK is set.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import holdout  # noqa: E402

ARR_PARQUET = ROOT / "data" / "bts" / "parquet_arr"
CACHE_DIR = ROOT / "data" / "cache" / "arrival"
AIRPORTS_REF = ROOT / "models" / "airports_ref.csv"
FIRST_MONTH = "2025-01"
LAST_MONTH = "2026-08"
UNLOCK_VAR = holdout.UNLOCK_VAR


def guard_month(ym: str):
    if UNLOCK_VAR in os.environ:
        sys.exit(f"{UNLOCK_VAR} is set; arrival scripts never run with it")
    if not (FIRST_MONTH <= ym <= LAST_MONTH):
        raise ValueError(f"{ym} is outside the protocol window {FIRST_MONTH}..{LAST_MONTH}")


def airport_tz() -> dict:
    ref = pd.read_csv(AIRPORTS_REF)
    return dict(zip(ref["iata"], ref["tz"]))


def _hhmm_to_min(s: pd.Series) -> np.ndarray:
    v = s.astype("float64").to_numpy()
    return np.where(np.isnan(v), np.nan, (np.nan_to_num(v) // 100) % 24 * 60 + np.nan_to_num(v) % 100)


def local_walls(df: pd.DataFrame, tz_of: dict) -> pd.DataFrame:
    """Wall-clock (naive, in minutes resolution) scheduled departure at the origin and scheduled arrival at the
    destination. Arrival date comes from scheduled departure -> UTC -> + CRSElapsedTime -> destination zone; its
    time of day is CRSArrTime (the legs' times are local to the hub, so no other conversion is needed). Rows
    with an unknown zone, missing CRSElapsedTime, or a nonexistent/ambiguous local time get NaT."""
    n = len(df)
    dep_wall = (df["FlightDate"].dt.normalize() + pd.to_timedelta(_hhmm_to_min(df["CRSDepTime"]), unit="m")).to_numpy()
    dep_wall = pd.Series(pd.to_datetime(dep_wall), index=df.index)
    elapsed = df["CRSElapsedTime"].astype("float64")
    arr_tod = pd.Series(_hhmm_to_min(df["CRSArrTime"]), index=df.index)
    arr_wall = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for o_tz in pd.unique(df["Origin"].map(tz_of).dropna()):
        o_mask = (df["Origin"].map(tz_of) == o_tz).to_numpy()
        dep_loc = dep_wall[o_mask].dt.tz_localize(o_tz, ambiguous="NaT", nonexistent="shift_forward")
        utc_arr = dep_loc.dt.tz_convert("UTC") + pd.to_timedelta(elapsed[o_mask], unit="m")
        idx = utc_arr.index
        for d_tz in pd.unique(df.loc[idx, "Dest"].map(tz_of).dropna()):
            sub = idx[(df.loc[idx, "Dest"].map(tz_of) == d_tz).to_numpy()]
            wall = utc_arr[sub].dt.tz_convert(d_tz).dt.tz_localize(None)
            base = wall.dt.normalize() + pd.to_timedelta(arr_tod[sub], unit="m")
            # snap to the day offset whose CRSArrTime is closest to the converted wall time
            cands = [base + pd.Timedelta(days=k) for k in (-1, 0, 1)]
            diffs = np.stack([(c - wall).abs().to_numpy() for c in cands])
            best = diffs.argmin(axis=0)
            pick = np.choose(best, [c.to_numpy() for c in cands])
            arr_wall[sub] = pd.to_datetime(pick)
    return pd.DataFrame({"dep_wall": dep_wall, "arr_wall": arr_wall}, index=df.index)


def arrival_label(df: pd.DataFrame) -> np.ndarray:
    """ArrDelay in minutes; NaN when cancelled; +inf when diverted (a diversion exceeds every threshold)."""
    out = df["ArrDelay"].astype("float64").to_numpy().copy()
    out[df["cancelled"].to_numpy()] = np.nan
    out[df["diverted"].to_numpy() & ~df["cancelled"].to_numpy()] = np.inf
    return out


def departure_label(df: pd.DataFrame) -> np.ndarray:
    out = df["DepDelay"].astype("float64").to_numpy().copy()
    out[df["cancelled"].to_numpy()] = np.nan
    return out


def build_cache_frame(df: pd.DataFrame, feats: pd.DataFrame, tz_of: dict) -> pd.DataFrame:
    """The per-month cache from a loaded frame (backtest.load_months output) and its committed features."""
    walls = local_walls(df, tz_of)
    out = feats.astype("float32").reset_index(drop=True).copy()
    meta = pd.DataFrame({
        "ym": df["ym"].to_numpy(),
        "flight_date": df["FlightDate"].to_numpy(),
        "network": df["Marketing_Airline_Network"].to_numpy(),
        "carrier": df["Reporting_Airline"].to_numpy(),
        "fn_marketing": df["Flight_Number_Marketing_Airline"].astype("float64").to_numpy(),
        "fn_operating": df["Flight_Number_Reporting_Airline"].astype("float64").to_numpy(),
        "tail": df["Tail_Number"].astype("object").fillna("").to_numpy(),
        "origin": df["Origin"].to_numpy(),
        "dest": df["Dest"].to_numpy(),
        "miles": df["Distance"].astype("float32").to_numpy(),
        "elapsed": df["CRSElapsedTime"].astype("float32").to_numpy(),
        "cancelled": df["cancelled"].to_numpy(),
        "diverted": df["diverted"].to_numpy(),
        "dep_delay": departure_label(df).astype("float32"),
        "arr_delay": arrival_label(df).astype("float32"),
        "dep_wall": walls["dep_wall"].to_numpy(),
        "arr_wall": walls["arr_wall"].to_numpy(),
    })
    return pd.concat([out, meta], axis=1)


def build_month(ym: str, force: bool = False, out_dir: Path = None) -> Path:
    guard_month(ym)
    out_dir = Path(out_dir or CACHE_DIR)
    out = out_dir / f"{ym}.parquet"
    if out.exists() and not force:
        return out
    import backtest as bt
    old = bt.PARQUET
    bt.PARQUET = ARR_PARQUET
    try:
        df = bt.load_months(ym, ym)
        assert df["FlightDate"].max() < holdout.LOCKED_FROM_TS, "a date on or after 2026-09 was loaded"
        feats = bt.committed_features(df)
    finally:
        bt.PARQUET = old
    cache = build_cache_frame(df, feats, airport_tz())
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(out, index=False, compression="zstd")
    return out


def load_cache(ym: str, cache_dir: Path = None) -> pd.DataFrame:
    guard_month(ym)
    return pd.read_parquet(Path(cache_dir or CACHE_DIR) / f"{ym}.parquet")


def months(start: str = FIRST_MONTH, end: str = LAST_MONTH) -> list[str]:
    return [f"{p.year}-{p.month:02d}" for p in pd.period_range(start, end, freq="M")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", nargs=2, metavar=("START", "END"), default=[FIRST_MONTH, LAST_MONTH])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    for ym in months(*args.months):
        p = build_month(ym, args.force)
        print(p.name, len(pd.read_parquet(p, columns=["ym"])), flush=True)


if __name__ == "__main__":
    main()
