"""
Build models/airports_ref.csv: iata,name,lat,lon,tz,state,daily_flights_2025.

Airport list = every origin/dest in BTS parquet_v2 2024-01..2026-08  U  the 30 airports in the app's
autocomplete (flyrely-app/src/lib/airports.ts). Names/coordinates/state from data/ref/airports.csv
(OurAirports); IANA time zones from `timezonefinder` (offline; `pip install timezonefinder` in the venv only,
it is not a runtime dependency). daily_flights_2025 = 2025 departures + arrivals / 365.

The 30 airports the API already had hand-typed coordinates for keep exactly those coordinates, so the
distance feature (and therefore predictions) for existing routes does not change.

    python scripts/build_airports_ref.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
from paths import BTS_PARQUET  # noqa: E402

REF = ROOT / "data" / "ref" / "airports.csv"
OUT = ROOT / "models" / "airports_ref.csv"
APP_AIRPORTS = ROOT.parent / "flyrely-app" / "src" / "lib" / "airports.ts"

LEGACY_COORDS = {
    "ATL": (33.6407, -84.4277), "ORD": (41.9742, -87.9073), "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737), "LAX": (33.9425, -118.4081), "JFK": (40.6413, -73.7781),
    "SFO": (37.6213, -122.3790), "SEA": (47.4502, -122.3088), "MIA": (25.7959, -80.2870),
    "PHX": (33.4373, -112.0078), "LAS": (36.0840, -115.1537), "MCO": (28.4312, -81.3081),
    "CLT": (35.2140, -80.9431), "EWR": (40.6895, -74.1745), "BOS": (42.3656, -71.0096),
    "MSP": (44.8848, -93.2223), "DTW": (42.2162, -83.3554), "PHL": (39.8744, -75.2424),
    "LGA": (40.7769, -73.8740), "DCA": (38.8512, -77.0402), "IAH": (29.9902, -95.3368),
    "SLC": (40.7899, -111.9791), "SAN": (32.7338, -117.1933), "TPA": (27.9756, -82.5333),
    "PDX": (45.5898, -122.5951), "BWI": (39.1774, -76.6684), "FLL": (26.0742, -80.1506),
    "MDW": (41.7868, -87.7522), "BNA": (36.1263, -86.6774), "AUS": (30.1975, -97.6664),
}


def bts_counts() -> tuple[set, pd.Series]:
    seen, y2025 = set(), pd.Series(dtype="float64")
    for p in sorted(BTS_PARQUET.glob("*.parquet")):
        if not "2024-01" <= p.stem <= "2026-08":
            continue
        df = pd.read_parquet(p, columns=["Origin", "Dest"])
        c = pd.concat([df["Origin"].astype(str), df["Dest"].astype(str)]).value_counts()
        seen |= set(c.index)
        if p.stem.startswith("2025-"):
            y2025 = y2025.add(c, fill_value=0)
    return seen, y2025


def main():
    from timezonefinder import TimezoneFinder
    codes, y2025 = bts_counts()
    codes |= set(re.findall(r"code:\s*'([A-Z]{3})'", APP_AIRPORTS.read_text()))
    ref = pd.read_csv(REF, usecols=["iata_code", "name", "latitude_deg", "longitude_deg", "iso_country",
                                    "iso_region", "type", "scheduled_service", "ident"])
    # Airports renamed in OurAirports (PBI is now listed under IATA "DJT") are found via their ICAO ident K<code>
    alias = ref[ref["ident"].str.fullmatch(r"K[A-Z]{3}") & (ref["scheduled_service"] == "yes")].copy()
    alias["iata_code"] = alias["ident"].str[1:]
    ref = pd.concat([ref.dropna(subset=["iata_code"]), alias[alias["iata_code"].isin(codes)]])
    ref["rank"] = (ref["scheduled_service"] == "yes").astype(int) * 10 + ref["type"].map(
        {"large_airport": 3, "medium_airport": 2, "small_airport": 1}).fillna(0)
    ref = ref.sort_values("rank", ascending=False).drop_duplicates("iata_code").set_index("iata_code")
    tf, rows, missing = TimezoneFinder(), [], []
    for c in sorted(codes):
        if c not in ref.index:
            missing.append(c)
            continue
        r = ref.loc[c]
        lat, lon = LEGACY_COORDS.get(c, (float(r["latitude_deg"]), float(r["longitude_deg"])))
        region = str(r["iso_region"])
        rows.append({"iata": c, "name": r["name"], "lat": round(lat, 4), "lon": round(lon, 4),
                     "tz": tf.timezone_at(lat=lat, lng=lon),
                     "state": region.split("-")[1] if region.startswith("US-") else "",
                     "daily_flights_2025": round(float(y2025.get(c, 0)) / 365, 1)})
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"{len(out)} airports -> {OUT}; not in OurAirports: {missing}; no tz: {out['tz'].isna().sum()}")


if __name__ == "__main__":
    main()
