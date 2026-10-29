"""
Build models/codeshare_map.csv: ticketed (marketing) flight -> operating carrier.

    sh: download BTS Marketing Carrier zips into data/bts/marketing/, e.g.
        https://transtats.bts.gov/PREZIP/On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_2026_8.zip
    python scripts/build_codeshare_map.py

One row per (marketing airline, marketing flight number, origin, dest), with the
operating airline and flight number seen most often on it and how often that was.
"""

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "bts" / "marketing"
OUT = ROOT / "models" / "codeshare_map.csv"

KEYS = ["marketing_airline", "marketing_flight_number", "origin", "dest"]
RENAME = {
    "Marketing_Airline_Network": "marketing_airline",
    "Flight_Number_Marketing_Airline": "marketing_flight_number",
    "Operating_Airline": "operating_airline",
    "Flight_Number_Operating_Airline": "operating_flight_number",
    "Origin": "origin",
    "Dest": "dest",
}


def read(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            # BTS pads some header names with a trailing space ("Operating_Airline ")
            df = pd.read_csv(f, usecols=lambda c: c.strip() in RENAME, dtype=str)
    return df.rename(columns=lambda c: RENAME[c.strip()])


def main():
    zips = sorted(SRC.glob("*.zip"))
    if not zips:
        raise SystemExit(f"no zips in {SRC}")
    df = pd.concat([read(z) for z in zips], ignore_index=True).dropna()
    counts = df.groupby(KEYS + ["operating_airline", "operating_flight_number"]).size().rename("n").reset_index()
    totals = counts.groupby(KEYS)["n"].transform("sum")
    counts["share"] = (counts["n"] / totals).round(3)
    best = counts.sort_values("n", ascending=False).drop_duplicates(KEYS)
    best = best.sort_values(KEYS)[KEYS + ["operating_airline", "operating_flight_number", "share", "n"]]
    best.to_csv(OUT, index=False)
    codeshare = best["marketing_airline"] != best["operating_airline"]
    print(f"{len(best):,} ticketed flights from {len(zips)} months; {codeshare.mean():.1%} operated by another carrier")
    print(best[codeshare].groupby(["marketing_airline", "operating_airline"]).size().sort_values(ascending=False).head(12))


if __name__ == "__main__":
    main()
