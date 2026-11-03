"""
Fetch archived weather *forecasts* (Open-Meteo Previous Runs API, free, no key)
for the busiest BTS airports, as they were issued 1 and 7 days ahead.

    python backtest/fetch_weather.py [top_n]   # default 100 airports

Writes one Parquet per airport to data/weather/ (UTC hourly). Skips airports
already fetched. Needs data/ref/airports.csv (OurAirports) for coordinates.
"""

import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import BTS_PARQUET as PARQUET  # noqa: E402  the one constant for the BTS data dir
OUT = ROOT / "data" / "weather"
AIRPORTS = ROOT / "data" / "ref" / "airports.csv"

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLES = ["precipitation", "snowfall", "wind_gusts_10m", "cape", "weather_code"]
LEADS = (1, 7)
START, END = "2025-01-01", "2026-08-31"  # fit, validation and holdout row months


def busiest_airports(n: int) -> list[str]:
    counts = pd.Series(dtype="int64")
    for p in sorted(PARQUET.glob("2025-*.parquet")) + sorted(PARQUET.glob("2026-*.parquet")):
        df = pd.read_parquet(p, columns=["Origin", "Dest"])
        c = pd.concat([df["Origin"].astype(str), df["Dest"].astype(str)]).value_counts()
        counts = counts.add(c, fill_value=0)
    return list(counts.sort_values(ascending=False).index[:n])


def fetch(client: httpx.Client, iata: str, lat: float, lon: float) -> pd.DataFrame:
    hourly = [f"{v}_previous_day{d}" for v in VARIABLES for d in LEADS]
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(hourly),
              "start_date": START, "end_date": END, "timezone": "GMT"}
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


def main(top_n: int):
    OUT.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(AIRPORTS, usecols=["iata_code", "latitude_deg", "longitude_deg", "iso_country"])
    ref = ref.dropna(subset=["iata_code"]).drop_duplicates("iata_code").set_index("iata_code")
    with httpx.Client(timeout=300) as client:
        for iata in busiest_airports(top_n):
            out = OUT / f"{iata}.parquet"
            if out.exists():
                continue
            if iata not in ref.index:
                print(f"{iata}: no coordinates, skipped")
                continue
            lat, lon = ref.loc[iata, ["latitude_deg", "longitude_deg"]]
            fetch(client, iata, float(lat), float(lon)).to_parquet(out, index=False)
            print(f"{iata} ok")
            time.sleep(8)  # stay well inside the free per-minute limit


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
