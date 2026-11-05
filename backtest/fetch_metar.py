"""
Fetch observed METARs (Iowa Environmental Mesonet ASOS archive, free, no key) for the busiest BTS airports.

    python backtest/fetch_metar.py [top_n] [--airports ATL,ANC] [--end 2026-08-31] [--pause 2.0]
    FLYRELY_HOLDOUT_UNLOCK=2026-09 python backtest/fetch_metar.py --holdout 2026-09

IEM guidance (asos.py?help): 1 s per-IP throttle (429 if exceeded, 503 if overloaded), 1,000 station-years per
request. So: strictly sequential, one request per station-year, a pause between requests (default 2 s),
exponential back-off on 429/503. Raw CSV is cached under data/metar/<IATA>_<year>.csv and never re-fetched
(resumable: an interrupted run continues where it stopped; files are written atomically).

Airports are BTS/IATA codes; the IEM station id comes from flyrely_features.stations (data/ref/airports.csv),
e.g. ANC -> PANC, SJU -> TJSJ, PBI -> DJT. Files stay named by the BTS code.

Holdout lock (backtest/holdout.py): dates after 2026-08-31 are always refused in the normal form. `--holdout YYYY-MM`
(with FLYRELY_HOLDOUT_UNLOCK=YYYY-MM) fetches that month plus one day, for the frozen airport list
(results/h1_airports.json), into data/holdout/YYYY-MM/metar/<IATA>_YYYY-MM.csv only.
"""

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import BTS_PARQUET as PARQUET  # noqa: E402
import holdout  # noqa: E402
sys.path.insert(0, str(ROOT))
from flyrely_features.stations import iem_station, load_table  # noqa: E402

OUT = ROOT / "data" / "metar"
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
FIELDS = ["vsby", "sknt", "gust", "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4",
          "wxcodes", "p01i", "metar"]
START_YEAR = 2024
LAST_ALLOWED = str(holdout.LAST_ALLOWED_DATE)     # 2026-08-31
UA = "flyrely-research/1.0 (sequential, cached; seanthomasbutton@gmail.com)"


def check_holdout(end: str, holdout_month: str = None) -> None:
    holdout.check_end_date(end, holdout_month)


def busiest_airports(n: int) -> list[str]:
    """The n busiest airports over 2025-01..2026-08. Months from 2026-09 on (the holdout) are never read."""
    counts = pd.Series(dtype="int64")
    for p in holdout.parquet_files_before_lock(PARQUET):
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


def fetch_one(client: httpx.Client, station: str, first: date, last: date, tries: int = 6,
              holdout_month: str = None) -> str:
    # IEM day2 is exclusive of the end date's last day only when hour is unset: ask up to day2+1 00:00
    p = params(station, first, last + timedelta(days=1))
    holdout.check_end_date(str(last), holdout_month, what="request end")
    for k in range(tries):
        try:
            r = client.get(URL, params=p, timeout=300)
        except httpx.TransportError as e:
            wait = min(600, 20 * 2 ** k)
            print(f"  {station} {first.year}: {e!r}, retry in {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.text
        wait = min(600, 20 * 2 ** k)
        print(f"  {station} {first.year}: HTTP {r.status_code}, back off {wait}s", flush=True)
        if r.status_code not in (429, 503, 502, 504):
            raise SystemExit(f"unexpected HTTP {r.status_code}: {r.text[:200]}")
        time.sleep(wait)
    raise SystemExit(f"giving up on {station} {first.year}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("top_n", nargs="?", type=int, default=100)
    ap.add_argument("--airports")
    ap.add_argument("--end", default=LAST_ALLOWED)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--holdout", metavar="YYYY-MM", help="fetch that protected month plus one day for the frozen "
                    "airport list, into data/holdout/YYYY-MM/metar/ (needs FLYRELY_HOLDOUT_UNLOCK=YYYY-MM)")
    a = ap.parse_args(argv)
    if a.holdout:
        if a.airports or a.end != LAST_ALLOWED or a.top_n != 100:
            raise SystemExit("--holdout takes no top_n, --airports or --end: the airports are the frozen list "
                             "and the dates are the month plus one day")
        month = holdout.require_unlock(a.holdout)
        first, last = holdout.input_window(month)
        check_holdout(str(last), month)
        airports = holdout.frozen_airports("metar")
        out_dir = holdout.holdout_dir(month) / "metar"
        todo = [(s, month, first, last) for s in airports if not (out_dir / f"{s}_{month}.csv").exists()]
    else:
        check_holdout(a.end)
        airports = a.airports.split(",") if a.airports else busiest_airports(a.top_n)
        out_dir = OUT
        todo = [(s, y, f, l) for s in airports for y, f, l in year_windows(a.end)
                if not (OUT / f"{s}_{y}.csv").exists()]
    out_dir.mkdir(parents=True, exist_ok=True)
    table = load_table()
    print(f"{len(airports)} airports, {len(todo)} station-years to fetch", flush=True)
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True) as client:
        for i, (s, y, f, l) in enumerate(todo, 1):
            text = fetch_one(client, iem_station(s, table), f, l, holdout_month=a.holdout)
            if not text.startswith("station,valid"):
                raise SystemExit(f"unexpected response for {s} {y}: {text[:200]}")
            tmp = out_dir / f"{s}_{y}.csv.part"
            tmp.write_text(text)
            tmp.rename(out_dir / f"{s}_{y}.csv")
            rows = text.count(chr(10)) - 1
            print(f"[{i}/{len(todo)}] {s} {y}: {rows} rows" + ("  WARNING: EMPTY, check the station id" if rows <= 0 else ""),
                  flush=True)
            time.sleep(a.pause)


if __name__ == "__main__":
    main()
