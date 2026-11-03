"""
Fetch archived weather *forecasts* (Open-Meteo Previous Runs API, free, no key)
for the busiest BTS airports, as they were issued 1 and 7 days ahead.

    python backtest/fetch_weather.py [top_n] [--airports ATL,ORD] [--end 2026-08-31]
                                              # default: busiest 100 in parquet_v2 + the app's 30

Writes one Parquet per airport to data/weather/ (UTC hourly). Skips airports
already fetched. Coordinates come from models/airports_ref.csv.
"""

import argparse
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import BTS_PARQUET as PARQUET  # noqa: E402  the one constant for the BTS data dir
OUT = ROOT / "data" / "weather"
REF = ROOT / "models" / "airports_ref.csv"

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLES = ["precipitation", "snowfall", "wind_gusts_10m", "cape", "weather_code"]
LEADS = (1, 7)
START, END = "2025-01-01", "2026-08-31"  # fit, validation and holdout row months


HOLDOUT_LAST_ALLOWED = "2026-08-31"  # months from 2026-09 on are the untouched holdout


def check_holdout(end: str) -> None:
    """Refuse to fetch weather into the holdout period unless FLYRELY_HOLDOUT_UNLOCK=1."""
    try:
        end_d = date.fromisoformat(end)
    except ValueError:
        raise SystemExit(f"invalid --end '{end}': expected YYYY-MM-DD")
    if end_d > date.fromisoformat(HOLDOUT_LAST_ALLOWED) and os.environ.get("FLYRELY_HOLDOUT_UNLOCK") != "1":
        raise SystemExit(f"refusing --end {end}: dates after {HOLDOUT_LAST_ALLOWED} are the untouched holdout "
                         "(set FLYRELY_HOLDOUT_UNLOCK=1 to override)")


def busiest_airports(n: int) -> list[str]:
    counts = pd.Series(dtype="int64")
    for p in sorted(PARQUET.glob("2025-*.parquet")) + sorted(PARQUET.glob("2026-*.parquet")):
        df = pd.read_parquet(p, columns=["Origin", "Dest"])
        c = pd.concat([df["Origin"].astype(str), df["Dest"].astype(str)]).value_counts()
        counts = counts.add(c, fill_value=0)
    return list(counts.sort_values(ascending=False).index[:n])


def fetch(client: httpx.Client, iata: str, lat: float, lon: float, end: str = END) -> pd.DataFrame:
    hourly = [f"{v}_previous_day{d}" for v in VARIABLES for d in LEADS]
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(hourly),
              "start_date": START, "end_date": end, "timezone": "GMT"}
    for attempt in range(8):
        r = client.get(URL, params=params)
        if r.status_code == 429:
            time.sleep(60 * (attempt + 1))
            continue
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame(h).rename(columns={"time": "utc_hour"})
        df["utc_hour"] = pd.to_datetime(df["utc_hour"])
        df.insert(0, "airport", iata)
        return df
    raise RuntimeError(f"{iata}: still rate-limited")


def app_airports() -> list[str]:
    """The airports the app's autocomplete offers (flyrely-app/src/lib/airports.ts), read-only."""
    ts = ROOT.parent / "flyrely-app" / "src" / "lib" / "airports.ts"
    return re.findall(r"code:\s*'([A-Z]{3})'", ts.read_text()) if ts.exists() else []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("top_n", nargs="?", type=int, default=100, help="busiest airports in parquet_v2 (default 100)")
    ap.add_argument("--airports", help="comma-separated IATA codes to fetch instead of the busiest-N list")
    ap.add_argument("--end", default=END, help=f"last date to fetch (default {END})")
    ap.add_argument("--budget-min", type=float, default=0, help="stop starting new airports after this many minutes (0 = no limit)")
    args = ap.parse_args(argv)
    check_holdout(args.end)

    OUT.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(REF).set_index("iata")  # models/airports_ref.csv (built from OurAirports)
    wanted = ([c.strip().upper() for c in args.airports.split(",")] if args.airports
              else list(dict.fromkeys(busiest_airports(args.top_n) + app_airports())))
    started, done, failed = time.time(), 0, {}
    with httpx.Client(timeout=300) as client:
        for iata in wanted:
            out = OUT / f"{iata}.parquet"
            if out.exists():
                continue
            if args.budget_min and (time.time() - started) / 60 > args.budget_min:
                print(f"time budget reached; resume with the same command ({len(wanted) - done} left)")
                break
            if iata not in ref.index:
                failed[iata] = "no coordinates in models/airports_ref.csv"
                print(f"{iata}: MISSING - {failed[iata]}")
                continue
            try:
                fetch(client, iata, float(ref.loc[iata, "lat"]), float(ref.loc[iata, "lon"]), args.end) \
                    .to_parquet(out, index=False)
            except Exception as e:  # keep going; report at the end
                failed[iata] = f"{type(e).__name__}: {e}"
                print(f"{iata}: MISSING - {failed[iata]}")
                continue
            done += 1
            print(f"{iata} ok")
            time.sleep(8)  # stay well inside the free per-minute limit
    if failed:
        print("not fetched:", failed)


if __name__ == "__main__":
    main()
