"""
Build data/bts/parquet_arr: every parquet_v2 column plus the arrival columns (ArrDelay, ArrTime, WheelsOn, TaxiIn,
ArrDel15, DivReachedDest, DivArrDelay), from the same Marketing Carrier zips as parquet_v2.

    python backtest/prepare_arrival.py [--force]

Protocol: backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md. Months 2026-09 and later are never converted: there is no holdout
flag, locked zips are skipped, and the script exits if FLYRELY_HOLDOUT_UNLOCK is set at all. prepare.py is imported
for its column spec and zip reader, not edited.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import holdout  # noqa: E402
import prepare  # noqa: E402

ROOT = prepare.ROOT
SRC = ROOT / "marketing"
OUT = ROOT / "parquet_arr"

ARRIVAL_COLUMNS = {
    "ArrDelay": "float32",
    "ArrTime": "Int16",
    "WheelsOn": "Int16",
    "TaxiIn": "float32",
    "ArrDel15": "float32",
    "DivReachedDest": "float32",
    "DivArrDelay": "float32",
}
COLUMNS = {**prepare.MARKETING_COLUMNS, **ARRIVAL_COLUMNS}


def _refuse_if_unlock_set():
    if holdout.UNLOCK_VAR in os.environ:
        sys.exit(f"{holdout.UNLOCK_VAR} is set; prepare_arrival.py never runs with it (holdout is not used here)")


def convert(zip_path: Path, out_dir: Path, force: bool = False) -> Path:
    _refuse_if_unlock_set()
    m = re.search(r"_(\d{4})_(\d{1,2})\.zip$", zip_path.name)
    if not m:
        raise SystemExit(f"cannot read the month from {zip_path.name}")
    ym = f"{int(m.group(1))}-{int(m.group(2)):02d}"
    if holdout.is_locked_month(ym):
        raise SystemExit(f"refusing {zip_path.name}: {ym} is in the protected holdout")
    out = out_dir / f"{ym}.parquet"
    if out.exists() and not force:
        return out
    df = prepare._read(zip_path, COLUMNS).rename(columns=prepare.MARKETING_RENAME)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    if len(df) and df["FlightDate"].max() >= holdout.LOCKED_FROM_TS:
        raise SystemExit(f"refusing {zip_path.name}: rows dated {holdout.LOCKED_FROM} or later")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild months that already exist")
    args = ap.parse_args(argv)
    _refuse_if_unlock_set()
    zips = sorted(SRC.glob("*.zip"))
    locked = [z for z in zips if holdout.is_locked_month(holdout.zip_month(z.name) or "0000-00")]
    if locked:
        print("skipping protected holdout zips:", [z.name for z in locked])
        zips = [z for z in zips if z not in locked]
    # the protocol covers 2025-01..2026-08 only
    zips = [z for z in zips if (holdout.zip_month(z.name) or "0000-00") >= "2025-01"]
    if not zips:
        sys.exit(f"no zips in {SRC}")
    for zp in zips:
        p = convert(zp, OUT, args.force)
        print(p.name, len(pd.read_parquet(p, columns=["Origin"])))


if __name__ == "__main__":
    main()
