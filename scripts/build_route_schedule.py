"""
Build models/route_schedule.csv.gz: the "typical schedule" behind the disruption copilot's alternatives.

    python scripts/build_route_schedule.py            # latest 8 weeks of data/bts/parquet_v2
    python scripts/build_route_schedule.py --weeks 8

Reads BTS Marketing Carrier data (data/bts/parquet_v2). The window is the 8 weeks (56 days) ending on the latest
flight date of the latest month on disk. Months in the protected holdout (2026-09 on) are never read: the script
stops at the last month before `holdout.LOCKED_FROM`, and never looks at FLYRELY_HOLDOUT_UNLOCK.

Columns (one row per marketing airline + marketing flight number + origin + dest + CRS departure time):
    marketing_airline         ticketed carrier, IATA (e.g. AA)
    marketing_flight_number   ticketed flight number, digits only (e.g. 117)
    origin, dest              IATA airports
    dep_time                  scheduled departure, local airport time, HH:MM (BTS CRSDepTime)
    arr_time                  scheduled arrival, local airport time, HH:MM (BTS CRSArrTime, most common value;
                              may fall on the next calendar day)
    operating_airline         carrier that flies it, most common value (e.g. OO for a SkyWest-operated flight)
    operating_flight_number   flight number at the operating carrier, most common value
    dow_mask                  weekdays the flight was seen on in the window, bit 0 = Monday ... bit 6 = Sunday
    days_seen                 distinct dates the flight was seen on in the window (rows with < 2 are dropped)
    schedule_as_of            latest month in the data, YYYY-MM (same value on every row)

Cancelled flights are not counted as seen (a cancelled flight is not a usable alternative that day).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
import holdout  # noqa: E402  read-only: the holdout guard
from paths import BTS_PARQUET  # noqa: E402

OUT = ROOT / "models" / "route_schedule.csv.gz"
MIN_DAYS_SEEN = 2
MAX_BYTES = 2 * 1024 * 1024
COLS = ["FlightDate", "Marketing_Airline_Network", "Flight_Number_Marketing_Airline", "Reporting_Airline",
        "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime", "CRSArrTime", "Cancelled"]
KEY = ["marketing_airline", "marketing_flight_number", "origin", "dest", "dep_time"]


def allowed_months(src: Path = BTS_PARQUET) -> list[str]:
    """Months on disk before the protected holdout, oldest first."""
    months = sorted(p.stem for p in src.glob("????-??.parquet"))
    return [m for m in months if holdout.is_month(m) and not holdout.is_locked_month(m)]


def hhmm(s: pd.Series) -> pd.Series:
    """BTS HHMM integer -> 'HH:MM' (2400 is midnight)."""
    v = s.astype(int) % 2400
    return (v // 100).astype(str).str.zfill(2) + ":" + (v % 100).astype(str).str.zfill(2)


def mode(s: pd.Series):
    return s.value_counts().index[0]


def load_window(weeks: int, src: Path = BTS_PARQUET, extra_src: Path = None) -> tuple[pd.DataFrame, str]:
    """`extra_src` is for scripts/refresh_schedule.py only: a second folder of month parquets that may hold months
    from the holdout on. The refresh script passes it only after the B5.1 holdout has been scored. Without it,
    holdout months are never read."""
    months = [(m, src) for m in allowed_months(src)]
    if extra_src is not None:
        have = {m for m, _ in months}
        months += [(p.stem, extra_src) for p in sorted(extra_src.glob("????-??.parquet"))
                   if holdout.is_month(p.stem) and p.stem not in have]
        months.sort()
    if not months:
        raise SystemExit(f"no BTS months before {holdout.LOCKED_FROM} in {src}")
    as_of = months[-1][0]
    need = max(2, weeks // 4 + 1)
    frames = []
    for ym, d in months[-need:]:
        assert d is not src or not holdout.is_locked_month(ym), ym
        frames.append(pd.read_parquet(d / f"{ym}.parquet", columns=COLS))
    df = pd.concat(frames, ignore_index=True)
    if extra_src is None:
        df = df[df["FlightDate"] < holdout.LOCKED_FROM_TS]
    end = df["FlightDate"].max()
    df = df[df["FlightDate"] > end - pd.Timedelta(days=7 * weeks)]
    return df, as_of


def build(df: pd.DataFrame, as_of: str) -> pd.DataFrame:
    # Spirit (NK) has ceased operating: its flights are not usable alternatives
    df = df[(df["Marketing_Airline_Network"] != "NK") & (df["Reporting_Airline"] != "NK")]
    df = df[df["CRSDepTime"].notna() & df["CRSArrTime"].notna() & df["Marketing_Airline_Network"].notna()
            & df["Flight_Number_Marketing_Airline"].notna() & (df["Cancelled"].fillna(0) == 0)]
    d = pd.DataFrame({
        "date": df["FlightDate"].dt.normalize().values,
        "dow": df["FlightDate"].dt.day_of_week.values,
        "marketing_airline": df["Marketing_Airline_Network"].astype(str).values,
        "marketing_flight_number": df["Flight_Number_Marketing_Airline"].astype(int).astype(str).values,
        "origin": df["Origin"].astype(str).values,
        "dest": df["Dest"].astype(str).values,
        "dep_time": hhmm(df["CRSDepTime"]).values,
        "arr_time": hhmm(df["CRSArrTime"]).values,
        "operating_airline": df["Reporting_Airline"].astype(str).values,
        "operating_flight_number": df["Flight_Number_Reporting_Airline"].astype("Int64").astype(str).values,
    })
    d = d.drop_duplicates(KEY + ["date"])
    g = d.groupby(KEY, sort=True)
    out = g.agg(days_seen=("date", "size"),
                arr_time=("arr_time", mode),
                operating_airline=("operating_airline", mode),
                operating_flight_number=("operating_flight_number", mode)).reset_index()
    bits = d.assign(bit=2 ** d["dow"]).drop_duplicates(KEY + ["dow"]).groupby(KEY)["bit"].sum().rename("dow_mask")
    out = out.merge(bits.reset_index(), on=KEY)
    out = out[out["days_seen"] >= MIN_DAYS_SEEN]
    out["schedule_as_of"] = as_of
    return out[KEY[:4] + ["dep_time", "arr_time", "operating_airline", "operating_flight_number",
                          "dow_mask", "days_seen", "schedule_as_of"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    df, as_of = load_window(a.weeks)   # holdout months are never read here; refresh_schedule.py passes extra_src
    start, end = df["FlightDate"].min().date(), df["FlightDate"].max().date()
    out = build(df, as_of)
    _write(out, a.out)   # fixed gzip mtime: the same data gives the same bytes
    size = a.out.stat().st_size
    routes = out.groupby(["origin", "dest"]).ngroups
    print(f"{len(out):,} flights, {routes:,} routes, window {start}..{end}, schedule_as_of {as_of}, "
          f"{size / 1024:.0f} KB gzipped")
    if size > MAX_BYTES:
        raise SystemExit(f"{a.out} is {size:,} bytes, over the {MAX_BYTES:,} byte limit")


def _write(out: pd.DataFrame, path: Path):
    import gzip
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(out.to_csv(index=False).encode())


if __name__ == "__main__":
    main()
