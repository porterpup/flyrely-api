"""arrival_data: local arrival walls (cross-midnight, DST), labels, window guard, feature parity with main."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
sys.path.insert(0, str(ROOT))
import arrival_data as ad  # noqa: E402

TZ = ad.airport_tz()


def row(origin, dest, date, dep, arr, elapsed):
    return {"FlightDate": pd.Timestamp(date), "Origin": origin, "Dest": dest, "CRSDepTime": dep, "CRSArrTime": arr,
            "CRSElapsedTime": float(elapsed)}


def walls(*rows):
    return ad.local_walls(pd.DataFrame(rows), TZ)


def test_cross_midnight_arrival_date():
    w = walls(row("SFO", "JFK", "2026-03-02", 2200, 600, 300))        # 22:00 PST + 5h = 08:00 EST... next day
    assert w.arr_wall[0] == pd.Timestamp("2026-03-03 06:00")           # CRSArrTime is the time of day, date from UTC path
    w = walls(row("LAX", "SEA", "2026-03-02", 2330, 130, 160))         # lands 01:30 next day local
    assert w.arr_wall[0] == pd.Timestamp("2026-03-03 01:30")
    assert w.dep_wall[0] == pd.Timestamp("2026-03-02 23:30")


def test_dst_start_day_dates():
    # LAX 23:00 PDT on 2026-03-08 (DST began 02:00 that day) + 300 min -> 07:00 EDT on 03-09
    w = walls(row("LAX", "JFK", "2026-03-08", 2300, 700, 300))
    assert w.arr_wall[0] == pd.Timestamp("2026-03-09 07:00")
    # JFK 01:30 EST + 120 min = 08:30 UTC = 03:30 CDT at ORD (CDT started 08:00 UTC)
    w = walls(row("JFK", "ORD", "2026-03-08", 130, 330, 120))
    assert w.arr_wall[0] == pd.Timestamp("2026-03-08 03:30")


def test_dst_end_day_and_unknowns():
    # ORD 23:00 CST 2026-11-01 (DST ended 02:00) + 120 -> 05:00 UTC... 23:00 CST = 05:00 UTC 11-02 ; +120 = 07:00 UTC = 02:00 CST
    w = walls(row("ORD", "DEN", "2026-11-01", 2300, 100, 120))
    assert w.arr_wall[0].date() == pd.Timestamp("2026-11-02").date()
    w = walls(row("ZZZ", "DEN", "2026-03-02", 800, 900, 60), row("ORD", "DEN", "2026-03-02", 800, 900, np.nan))
    assert w.arr_wall.isna().all()


def test_labels_diverted_inf_cancelled_nan():
    df = pd.DataFrame({"ArrDelay": [5.0, np.nan, np.nan], "DepDelay": [3.0, np.nan, 10.0],
                       "cancelled": [False, True, False], "diverted": [False, False, True]})
    a = ad.arrival_label(df)
    assert a[0] == 5.0 and np.isnan(a[1]) and np.isposinf(a[2])
    d = ad.departure_label(df)
    assert d[0] == 3.0 and np.isnan(d[1]) and d[2] == 10.0


def test_window_guard(monkeypatch):
    ad.guard_month("2025-01")
    ad.guard_month("2026-08")
    for bad in ("2024-12", "2026-09", "2027-01"):
        with pytest.raises(ValueError):
            ad.guard_month(bad)
    monkeypatch.setenv("FLYRELY_HOLDOUT_UNLOCK", "2026-09")
    with pytest.raises(SystemExit):
        ad.guard_month("2026-01")


def test_cached_features_equal_main_build_features():
    """500 rows through the same committed_features code the cache uses vs main.build_features row by row."""
    import backtest as bt
    import main
    rng = np.random.default_rng(1)
    aps = [a for a in ["ATL", "ORD", "DEN", "LAX", "JFK", "SEA", "MIA", "BOS", "SFO", "XXX"]]
    carriers = list(main.AIRLINE_DELAY_RATES)[:6]
    n = 500
    dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 240, n), unit="D")
    o = rng.choice(aps, n)
    d = np.array([rng.choice([a for a in aps if a != x]) for x in o])
    df = pd.DataFrame({"FlightDate": dates, "Origin": o, "Dest": d, "Reporting_Airline": rng.choice(carriers, n),
                       "dep_hour": rng.integers(0, 24, n)})
    df["dow"] = df["FlightDate"].dt.dayofweek + 1
    df["month"] = df["FlightDate"].dt.month
    f = bt.committed_features(df)
    assert bt.check_parity(df, f, n=500) == 0
    cached = f.astype("float32")
    for i in rng.choice(n, 50, replace=False):
        r = df.iloc[i]
        ref = main.build_features(r["Origin"], r["Dest"], r["FlightDate"].to_pydatetime().replace(hour=int(r["dep_hour"])),
                                  r["Reporting_Airline"])
        assert all(np.isclose(float(ref[k]), float(cached.iloc[i][k]), rtol=1e-5) for k in cached.columns)
