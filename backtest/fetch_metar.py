"""
Fetch observed METARs (Iowa Environmental Mesonet ASOS archive, free, no key) for the busiest BTS airports.

    python backtest/fetch_metar.py [top_n] [--airports ATL,ORD] [--end 2026-08-31] [--pause 2.0]

IEM guidance (asos.py?help): 1 s per-IP throttle (429 if exceeded, 503 if overloaded), 1,000 station-years per
request. So: strictly sequential, one request per station-year, a pause between requests (default 2 s),
exponential back-off on 429/503. Raw CSV is cached under data/metar/<IATA>_<year>.csv and never re-fetched
(resumable: an interrupted run continues where it stopped; files are written atomically).

Holdout lock: dates after 2026-08-31 are refused unless FLYRELY_HOLDOUT_UNLOCK=1.
"""

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import BTS_PARQUET as PARQUET  # noqa: E402

OUT = ROOT / "data" / "metar"
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
FIELDS = ["vsby", "sknt", "gust", "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4",
          "wxcodes", "p01i", "metar"]
START_YEAR = 2024
HOLDOUT_LAST_ALLOWED = "2026-08-31"
UA = "flyrely-research/1.0 (sequential, cached; seanthomasbutton@gmail.com)"


def check_holdout(end: str) -> None:
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
        counts = counts.add(pd.concat([df["Origin"].astype(str), df["Dest"].astype(str)]).value_counts(), fill_value=0)
    return list(counts.sort_values(ascending=False).index[:n])


def year_windows(end: str):
    end_d = date.fromisoformat(end)
    for y in range(START_YEAR, end_d.year + 1):
        last = min(end_d, date(y, 12, 31))
        yield y, date(y, 1, 1), last


def params(station: str, first: date, last: date) -> dict:
    return {"station": station, "data": FIELDS, "year1": first.year, "month1": first.month, "day1": first.day,
            "year2": last.year, "month2": last.month, "day2": last.day, "tz": "Etc/UTC", "format": "onlycomma",
            "latlon": "no", "missing": "empty", "trace": "0.0001", "report_type": [3, 4]}


def fetch_one(client: httpx.Client, station: str, first: date, last: date, tries: int = 6) -> str:
    # IEM day2 is exclusive of the end date's last day only when hour is unset: ask up to day2+1 00:00
    p = params(station, first, last + timedelta(days=1))
    if last + timedelta(days=1) > date.fromisoformat(HOLDOUT_LAST_ALLOWED) + timedelta(days=1) and \
            os.environ.get("FLYRELY_HOLDOUT_UNLOCK") != "1":
        raise SystemExit("holdout lock")
    for k in range(tries):
        r = client.get(URL, params=p, timeout=300)
        if r.status_code == 200:
            return r.text
        wait = min(600, 20 * 2 ** k)
        print(f"  {station} {first.year}: HTTP {r.status_code}, back off {wait}s", flush=True)
        if r.status_code not in (429, 503, 502, 504):
            raise SystemExit(f"unexpected HTTP {r.status_code}: {r.text[:200]}")
        time.sleep(wait)
    raise SystemExit(f"giving up on {station} {first.year}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("top_n", nargs="?", type=int, default=100)
    ap.add_argument("--airports")
    ap.add_argument("--end", default=HOLDOUT_LAST_ALLOWED)
    ap.add_argument("--pause", type=float, default=2.0)
    a = ap.parse_args()
    check_holdout(a.end)
    airports = a.airports.split(",") if a.airports else busiest_airports(a.top_n)
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [(s, y, f, l) for s in airports for y, f, l in year_windows(a.end)
            if not (OUT / f"{s}_{y}.csv").exists()]
    print(f"{len(airports)} airports, {len(todo)} station-years to fetch", flush=True)
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True) as client:
        for i, (s, y, f, l) in enumerate(todo, 1):
            text = fetch_one(client, s, f, l)
            if not text.startswith("station,valid"):
                raise SystemExit(f"unexpected response for {s} {y}: {text[:200]}")
            tmp = OUT / f"{s}_{y}.csv.part"
            tmp.write_text(text)
            tmp.rename(OUT / f"{s}_{y}.csv")
            print(f"[{i}/{len(todo)}] {s} {y}: {text.count(chr(10)) - 1} rows", flush=True)
            time.sleep(a.pause)


if __name__ == "__main__":
    main()
