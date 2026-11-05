"""
Convert BTS monthly zips into slim per-month Parquet.

    python backtest/prepare.py                      # Reporting Carrier zips (data/bts/raw) -> data/bts/parquet
    python backtest/prepare.py --source marketing   # Marketing Carrier zips (data/bts/marketing) -> data/bts/parquet_v2
    FLYRELY_HOLDOUT_UNLOCK=2026-09 python backtest/prepare.py --source marketing --holdout 2026-09
                                                    # the one holdout zip -> data/holdout/2026-09/bts/2026-09.parquet

The Reporting Carrier file omits six operating carriers (9E, PT, QX, G7, C5, YV; ~9.6% of
flights). The Marketing Carrier file has all of them, so the backtests read the --source
marketing output (see backtest/paths.py).
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import holdout  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "bts"

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

# Marketing Carrier files: source column (after stripping the trailing spaces BTS adds to
# some headers) -> output column. The operating carrier becomes the "reporting" carrier so
# downstream code is unchanged.
MARKETING_RENAME = {
    "Operating_Airline": "Reporting_Airline",
    "Flight_Number_Operating_Airline": "Flight_Number_Reporting_Airline",
}
MARKETING_COLUMNS = {
    **{k: v for k, v in COLUMNS.items() if k not in MARKETING_RENAME.values()},
    "Operating_Airline": "category",
    "Flight_Number_Operating_Airline": "Int32",
    "Flight_Number_Marketing_Airline": "Int32",
    "WheelsOff": "Int16",
    "TaxiOut": "float32",
}


def _read(zip_path: Path, columns: dict) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            raw_header = pd.read_csv(f, nrows=0).columns
        stripped = {c: c.strip() for c in raw_header}
        dtypes = {raw: columns[s] for raw, s in stripped.items() if s in columns}
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=list(dtypes), dtype=dtypes)
    return df.rename(columns=stripped)


def convert(zip_path: Path, out_dir: Path, source: str = "reporting", force: bool = False,
            holdout_month: str = None) -> Path:
    """`holdout_month`: the one unlocked month (checked by the caller); only that month's zip converts, and only
    rows of that month are written."""
    y, m = map(int, re.search(r"_(\d{4})_(\d{1,2})\.zip$", zip_path.name).groups())
    ym = f"{y}-{m:02d}"
    if holdout_month is None and holdout.is_locked_month(ym):
        raise SystemExit(f"refusing {zip_path.name}: {ym} is in the protected holdout (use --holdout {ym})")
    if holdout_month is not None and ym != holdout_month:
        raise SystemExit(f"refusing {zip_path.name}: --holdout {holdout_month} converts that month only")
    out = out_dir / f"{ym}.parquet"
    if out.exists() and not force:
        return out
    if source == "marketing":
        df = _read(zip_path, MARKETING_COLUMNS).rename(columns=MARKETING_RENAME)
    else:
        df = _read(zip_path, COLUMNS)
        # The Reporting Carrier files have no marketing airline; fall back to the operating one
        if "Marketing_Airline_Network" not in df:
            df["Marketing_Airline_Network"] = df["Reporting_Airline"]
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    if holdout_month is not None:
        holdout.refuse_rows_outside(df, holdout_month)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["reporting", "marketing"], default="reporting")
    ap.add_argument("--force", action="store_true", help="rebuild months that already exist")
    ap.add_argument("--holdout", metavar="YYYY-MM", help="convert only that protected month's zip, from and to "
                    "data/holdout/YYYY-MM/bts/ (needs FLYRELY_HOLDOUT_UNLOCK=YYYY-MM and --source marketing)")
    args = ap.parse_args(argv)
    if args.holdout:
        if args.source != "marketing":
            sys.exit("--holdout works with --source marketing only")
        month = holdout.require_unlock(args.holdout)
        d = holdout.holdout_dir(month) / "bts"
        zips = [z for z in sorted(d.glob("*.zip")) if holdout.zip_month(z.name) == month]
        if not zips:
            sys.exit(f"no {month} zip in {d} - run fetch_bts.sh --holdout {month} first")
        for zp in zips:
            p = convert(zp, d, "marketing", args.force, holdout_month=month)
            print(p.name, len(pd.read_parquet(p, columns=["Origin"])))
        return
    src, out_dir = (ROOT / "marketing", ROOT / "parquet_v2") if args.source == "marketing" \
        else (ROOT / "raw", ROOT / "parquet")
    zips = sorted(src.glob("*.zip"))
    locked = [z for z in zips if holdout.is_locked_month(holdout.zip_month(z.name) or "0000-00")]
    if locked:
        print("skipping protected holdout zips (convert them with --holdout):", [z.name for z in locked])
        zips = [z for z in zips if z not in locked]
    if not zips:
        sys.exit(f"no zips in {src} — run backtest/fetch_bts.sh first")
    for zp in zips:
        p = convert(zp, out_dir, args.source, args.force)
        print(p.name, len(pd.read_parquet(p, columns=["Origin"])))


if __name__ == "__main__":
    main()
