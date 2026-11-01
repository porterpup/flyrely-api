"""Convert BTS monthly zips (data/bts/raw) into slim per-month Parquet (data/bts/parquet)."""

import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data" / "bts"
RAW, OUT = ROOT / "raw", ROOT / "parquet"

COLUMNS = {
    "FlightDate": "string",
    "Reporting_Airline": "category",
    "Marketing_Airline_Network": "category",
    "Tail_Number": "string",
    "Flight_Number_Reporting_Airline": "Int32",
    "Origin": "category",
    "Dest": "category",
    "CRSDepTime": "Int16",
    "CRSArrTime": "Int16",
    "CRSElapsedTime": "float32",
    "DepDelay": "float32",
    "Cancelled": "float32",
    "CancellationCode": "category",
    "Diverted": "float32",
    "Distance": "float32",
}


def convert(zip_path: Path) -> Path:
    y, m = map(int, re.search(r"_(\d{4})_(\d{1,2})\.zip$", zip_path.name).groups())
    out = OUT / f"{y}-{m:02d}.parquet"
    if out.exists():
        return out
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            # The Reporting Carrier files have no marketing airline; fall back to the operating one
            df = pd.read_csv(f, usecols=lambda c: c in COLUMNS, dtype=COLUMNS)
    if "Marketing_Airline_Network" not in df:
        df["Marketing_Airline_Network"] = df["Reporting_Airline"]
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    zips = sorted(RAW.glob("*.zip"))
    if not zips:
        sys.exit("no zips in data/bts/raw — run backtest/fetch_bts.sh first")
    for zp in zips:
        p = convert(zp)
        print(p.name, len(pd.read_parquet(p, columns=["Origin"])))
