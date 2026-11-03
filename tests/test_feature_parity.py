"""Training/serving parity: the vectorised feature code and serve_row must agree on every sampled row.

The synthetic fixture needs no data/ directory. A second, data-backed version runs on real BTS months
when data/bts is present.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flyrely_features import booking, network, recent, rotation, weather  # noqa: E402
from flyrely_features._common import DAY0, HORIZONS, SHIFT, Counter, build_counters  # noqa: E402

N_SAMPLE = 2000
N_BOOKING_SERVE = 400  # booking serve_row does ~14 pandas merges per call; keep CI fast (full 2000 in the data-backed test)
AIRPORTS = ["ATL", "ORD", "DEN", "LAX", "JFK", "SEA", "MIA", "BOS"]
CARRIERS = ["DL", "AA", "UA", "WN"]


def same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.array_equal(a, b, equal_nan=True) or np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-9)


def synthetic_schedule(n=6000, seed=0):
    """Arrays shaped like runup.load() produces, for n flights over 40 days."""
    rng = np.random.default_rng(seed)
    o = rng.integers(0, len(AIRPORTS), n)
    d = (o + rng.integers(1, len(AIRPORTS), n)) % len(AIRPORTS)
    day = rng.integers(0, 40, n)
    dep_local = rng.integers(5 * 60, 23 * 60, n).astype(float)
    elapsed = rng.integers(60, 300, n).astype(float)
    s_dep = day * 1440 + dep_local + SHIFT
    cancelled = rng.random(n) < 0.03
    dep_delay = np.where(rng.random(n) < 0.25, rng.exponential(40, n), rng.integers(-10, 5, n)).astype(float)
    dep_delay[cancelled] = np.nan
    tail = rng.integers(0, 120, n)
    tail[rng.random(n) < 0.05] = -1
    A = {"o_id": o, "d_id": d, "al_id": rng.integers(0, len(CARRIERS), n), "day": day, "dep_local": dep_local,
         "s_dep": s_dep, "s_arr": s_dep + elapsed, "elapsed": elapsed, "cancelled": cancelled,
         "dep_delay": dep_delay, "tail_id": tail, "utc_shift": np.full(n, 300.0)}
    A["rt_id"] = A["o_id"] * len(AIRPORTS) + A["d_id"]
    A["fn_id"] = rng.integers(0, 300, n)
    A["a_dep"] = np.where(cancelled, np.nan, s_dep + np.nan_to_num(dep_delay))
    A["d15"] = (np.nan_to_num(dep_delay) >= 15) & ~cancelled
    return A


@pytest.fixture(scope="module")
def sched():
    A = synthetic_schedule()
    return A, build_counters(A), rotation.Rotation(A), np.random.default_rng(1).choice(len(A["day"]), N_SAMPLE, replace=False)


def query(A, i, keys):
    return {k: A[k][[i]] for k in keys}


@pytest.mark.parametrize("h", list(HORIZONS))
def test_recent_parity(sched, h):
    A, C, _, rows = sched
    vec = recent.recent_features(A, C, rows, h)
    keys = ["day", "dep_local", "o_id", "d_id", "al_id", "rt_id", "fn_id"]
    bad = 0
    for j, i in enumerate(rows):
        s = recent.serve_row(query(A, i, keys), C, h)
        bad += sum(not same(vec[k][j], s[k]) for k in vec)
    assert bad == 0


@pytest.mark.parametrize("h", ["3h", "1h"])
def test_network_parity(sched, h):
    A, C, _, rows = sched
    c = A["s_dep"][rows] - HORIZONS[h]
    vec = network.network_features(A, C, rows, c)
    bad = 0
    for j, i in enumerate(rows):
        s = network.serve_row(query(A, i, ["o_id", "d_id", "al_id"]), C, c[j])
        bad += sum(not same(vec[k][j], s[k]) for k in vec)
    assert bad == 0


@pytest.mark.parametrize("h", ["3h", "1h"])
def test_rotation_parity(sched, h):
    A, _, R, rows = sched
    c = A["s_dep"][rows] - HORIZONS[h]
    vec = rotation.rotation_features(A, R, rows, c)
    assert vec["rot_has_leg"].sum() > 0  # the fixture actually exercises the rotation path
    bad = 0
    for j, i in enumerate(rows):
        s = rotation.serve_row(A, R, i, c[j])
        bad += sum(not same(vec[k][j], s[k]) for k in vec)
    assert bad == 0


def test_weather_parity(tmp_path):
    A = synthetic_schedule()
    airports = pd.Index(sorted(AIRPORTS))
    hours = pd.date_range(DAY0, periods=45 * 24, freq="h")
    rng = np.random.default_rng(2)
    frames = []
    for ap in airports:
        f = pd.DataFrame({"airport": ap, "utc_hour": hours})
        for lead in (1, 7):
            for v in ("precipitation", "snowfall", "wind_gusts_10m", "cape", "weather_code"):
                f[f"{v}_previous_day{lead}"] = rng.random(len(hours))
        frames.append(f)
    pd.concat(frames).to_parquet(tmp_path / "wx.parquet")
    W = weather.Weather(airports, tmp_path)
    assert W.ok
    rows = np.random.default_rng(3).choice(len(A["day"]), N_SAMPLE, replace=False)
    keys = ["o_id", "d_id", "s_dep", "s_arr", "utc_shift"]
    for lead in (1, 7):
        vec = W.features(A, rows, lead)
        assert not np.isnan(vec["wx_o_precipitation"]).all()
        bad = 0
        for j, i in enumerate(rows):
            s = weather.serve_row(W, query(A, i, keys), lead)
            bad += sum(not same(vec[k][j], s[k]) for k in vec)
        assert bad == 0


def synthetic_flights(n, seed, start="2025-01-01", days=60):
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    df = pd.DataFrame({
        "FlightDate": dates, "Reporting_Airline": rng.choice(CARRIERS, n), "Origin": rng.choice(AIRPORTS, n),
        "dep_hour": rng.integers(5, 23, n), "Distance": rng.integers(200, 2500, n).astype(float),
        "CRSElapsedTime": rng.integers(60, 300, n).astype(float),
    })
    df["Dest"] = rng.choice(AIRPORTS, n)
    df["dow"] = df["FlightDate"].dt.dayofweek + 1
    df["month"] = df["FlightDate"].dt.month
    df["cancelled"] = rng.random(n) < 0.02
    df["delayed"] = np.where(df["cancelled"], np.nan, (rng.random(n) < 0.2).astype(float))
    return df


def booking_parity(df, hist, n_serve):
    tables = booking.BookingTables(hist)
    vec = booking.booking_features(df, hist)
    assert list(vec.columns) == list(tables.features(df).columns)
    pd.testing.assert_frame_equal(vec, tables.features(df))
    bad = 0
    for i in np.random.default_rng(4).choice(len(df), n_serve, replace=False):
        s = booking.serve_row(df.iloc[i], tables)
        bad += sum(not same(vec.iloc[i][k], s[k]) for k in vec.columns)
    return bad


def test_booking_parity_synthetic():
    hist = synthetic_flights(8000, seed=5, start="2024-06-01", days=200)
    df = synthetic_flights(N_SAMPLE, seed=6)
    df.loc[df.index[:20], "Origin"] = "ZZZ"  # unseen group falls back to the global rate
    assert booking_parity(df, hist, N_BOOKING_SERVE) == 0


def test_smoothed_rate_shrinks_toward_prior():
    hist = pd.DataFrame({"k": ["a"] * 10, "y": [1.0] * 10})
    r = booking.smoothed_rate(hist, ["k"], "y", 0.2)
    assert r.iloc[0, 1] == pytest.approx((10 + 50 * 0.2) / (10 + 50))


# --- data-backed (skipped in CI, where data/ does not exist) ----------------------------------

BTS = ROOT / "data" / "bts"


@pytest.mark.skipif(not BTS.exists(), reason="data/bts not present")
def test_booking_parity_real_data():
    sys.path.insert(0, str(ROOT / "backtest"))
    import backtest as bt
    allm = bt.load_months("2025-01", "2025-03")
    hist, df = allm[allm["ym"] <= "2025-02"], allm[allm["ym"] == "2025-03"].sample(N_SAMPLE, random_state=0)
    assert booking_parity(df, hist, N_SAMPLE) == 0
