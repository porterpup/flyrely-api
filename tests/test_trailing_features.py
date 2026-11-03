"""Trailing-rate features (B1): vectorised vs serve_row parity, leakage with a data gap, level shift, lagged route stats."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flyrely_features import booking, recent  # noqa: E402

GROUPS = recent.TRAIL_GROUPS + ("g",)


def synth(n=8000, days=200, seed=0):
    rng = np.random.default_rng(seed)
    A = {"day": rng.integers(0, days, n), "cancelled": rng.random(n) < 0.03}
    A["o_id"], A["d_id"] = rng.integers(0, 6, n), rng.integers(0, 6, n)
    A["al_id"], A["nw_id"] = rng.integers(0, 4, n), rng.integers(0, 3, n)
    A["rt_id"] = A["o_id"] * 6 + A["d_id"]
    A["oh_id"] = A["o_id"] * 24 + rng.integers(0, 24, n)
    A["fn_id"] = rng.integers(0, 60, n)
    A["fn_id"][rng.random(n) < 0.1] = -1
    A["g_id"] = np.zeros(n, dtype=int)
    A["d15"] = (rng.random(n) < 0.25) & ~A["cancelled"]
    return A


def sub(A, mask):
    return {k: v[mask] for k, v in A.items()}


def same(a, b):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True, rtol=0, atol=1e-9)


@pytest.mark.parametrize("gap", [1, 90])
def test_trailing_serve_row_parity(gap):
    A = synth()
    C = recent.daily_counters(A)
    rows = np.random.default_rng(1).choice(len(A["day"]), 300, replace=False)
    vec = recent.trailing_features(A, C, rows, gap)
    assert set(vec) == set(recent.trail_names())
    for j, i in enumerate(rows):
        s = recent.trailing_serve_row({k: A[k][[i]] for k in ["day"] + [f"{n}_id" for n in GROUPS]}, C, gap)
        assert all(same(vec[k][j], s[k]) for k in vec)


@pytest.mark.parametrize("gap", [1, 31, 90, 120])
def test_no_feature_uses_a_day_after_departure_minus_gap(gap):
    """Rebuild the counters from only days <= D-gap; the row's features must not change, and outcomes after must not matter."""
    A = synth(seed=3)
    C_full = recent.daily_counters(A)
    rows = np.random.default_rng(2).choice(len(A["day"]), 60, replace=False)
    full = recent.trailing_features(A, C_full, rows, gap)
    for i in rows:
        D = A["day"][i]
        C_cut = recent.daily_counters(sub(A, A["day"] <= D - gap))
        q = {k: A[k][[i]] for k in ["day"] + [f"{n}_id" for n in GROUPS]}
        cut = recent.trailing_serve_row(q, C_cut, gap)
        assert all(same(full[k][list(rows).index(i)], cut[k]) for k in cut)
        # flipping every outcome after the cutoff changes nothing
        B = {k: v.copy() for k, v in A.items()}
        late = B["day"] > D - gap
        B["d15"][late] = ~B["d15"][late] & ~B["cancelled"][late]
        B["cancelled"][late] = ~B["cancelled"][late]
        flipped = recent.trailing_serve_row(q, recent.daily_counters(B), gap)
        assert all(same(flipped[k], cut[k]) for k in cut)


def test_gap_changes_values_and_smoothing_falls_back_to_parent():
    A = synth()
    C = recent.daily_counters(A)
    rows = np.arange(500)
    a, b = recent.trailing_features(A, C, rows, 1), recent.trailing_features(A, C, rows, 90)
    assert not same(a["t30_o_delay"], b["t30_o_delay"])
    q = {k: np.array([v]) for k, v in {"day": 5, "o_id": 0, "d_id": 0, "al_id": 0, "nw_id": 0, "rt_id": 0, "oh_id": 0, "fn_id": 0, "g_id": 0}.items()}
    empty = recent.trailing_serve_row(q, C, 120)  # window entirely before the data: no history at all
    assert all(np.isnan(v) for v in empty.values())


def test_missing_flight_number_is_nan_and_blanking():
    A = synth()
    C = recent.daily_counters(A)
    rows = np.flatnonzero(A["fn_id"] < 0)
    out = recent.trailing_features(A, C, rows[A["day"][rows] >= 100], 30)
    assert np.isnan(out["t30_fn_delay"]).all() and not np.isnan(out["t30_o_delay"]).any()
    late = np.flatnonzero(A["day"] >= 100)[:100]
    X = pd.DataFrame(recent.trailing_features(A, C, late, 30))
    mask = np.arange(100) < 30
    recent.blank_flight_number(X, mask)
    assert X.loc[mask, [c for c in X if "_fn_" in c]].isna().all().all()
    assert X.loc[~mask, "t30_fn_delay"].notna().any() and X["t30_o_delay"].notna().all()


def test_level_shift():
    trail = {"t90_al_delay": np.array([0.3]), "t90_al_cancel": np.array([0.02]),
             "t90_o_delay": np.array([0.25]), "t90_o_cancel": np.array([0.01])}
    lr = {"rate_Reporting_Airline_delayed": np.array([0.2]), "rate_Reporting_Airline_c": np.array([0.01]),
          "rate_Origin_delayed": np.array([0.2]), "rate_Origin_c": np.array([0.01])}
    s = recent.level_shift_features(trail, lr)
    assert np.isclose(s["shift_al_delay"][0], 0.1) and np.isclose(s["shift_o_delay"][0], 0.05)


def flights(n, seed, start="2024-03-01", days=60):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"FlightDate": pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D"),
                       "Reporting_Airline": rng.choice(["DL", "AA", "UA"], n), "Origin": rng.choice(["ATL", "ORD", "DEN"], n),
                       "Dest": rng.choice(["LAX", "JFK", "SEA"], n), "dep_hour": rng.integers(5, 23, n)})
    df["Distance"] = df["Origin"].map({"ATL": 1000, "ORD": 1500, "DEN": 900}) + df["Dest"].map({"LAX": 100, "JFK": 200, "SEA": 300})
    df["CRSElapsedTime"] = df["Distance"] / 8
    df["dow"], df["month"] = df["FlightDate"].dt.dayofweek + 1, df["FlightDate"].dt.month
    df["cancelled"] = rng.random(n) < 0.02
    df["delayed"] = np.where(df["cancelled"], np.nan, (rng.random(n) < 0.2).astype(float))
    return df


def test_lagged_route_stats_ignore_the_row_and_serve_row_matches():
    hist = flights(4000, 0)
    df = flights(300, 1, start="2024-06-01")
    tables = booking.BookingTables(hist, lagged_route_stats=True)
    X = tables.features(df)
    df2 = df.assign(Distance=99999.0, CRSElapsedTime=1.0)  # the row's own values must not matter
    assert same(X["Distance"], tables.features(df2)["Distance"])
    expected = hist.groupby(["Origin", "Dest", "Reporting_Airline"])["Distance"].median()
    r = df.iloc[0]
    assert X["Distance"].iloc[0] == expected[(r["Origin"], r["Dest"], r["Reporting_Airline"])]
    for i in range(0, 300, 30):
        row = df.iloc[i].drop(["Distance", "CRSElapsedTime"])
        assert booking.serve_row(row, tables).equals(X.iloc[i].astype(float).rename(i)) or same(booking.serve_row(row, tables), X.iloc[i])
    unseen = df.iloc[[0]].assign(Origin="XXX")
    assert np.isnan(tables.features(unseen)["Distance"].iloc[0])


def test_unseen_group_falls_back_to_parent_rate():
    """A group with no flights in the window gets exactly its parent's rate, down the chain fn -> rt -> (o, d) -> global."""
    A = synth()
    C = recent.daily_counters(A)
    day = 150
    win = (A["day"] > day - 1 - 30) & (A["day"] <= day - 1)
    flown = win & ~A["cancelled"]
    g_delay = A["d15"][flown].sum() / flown.sum()
    g_cancel = A["cancelled"][win].sum() / win.sum()
    q = {k: np.array([v]) for k, v in {"day": day, "o_id": 77, "d_id": 78, "al_id": 0, "nw_id": 0, "rt_id": 7777,
                                      "oh_id": 7778, "fn_id": 7779, "g_id": 0}.items()}
    q["al_id"], q["nw_id"] = np.array([55]), np.array([56])  # unseen carrier / network too
    out = recent.trailing_serve_row(q, C, 1)
    for n in ("o", "d", "al", "nw", "rt", "oh", "fn"):
        assert np.isclose(out[f"t30_{n}_delay"], g_delay) and np.isclose(out[f"t30_{n}_cancel"], g_cancel)
    # a partially sparse group moves only part of the way from its parent (smoothing, not replacement)
    seen = {k: A[k][[0]] for k in ["day", "o_id", "d_id", "al_id", "nw_id", "rt_id", "oh_id", "fn_id", "g_id"]}
    seen["day"] = np.array([day])
    r = recent.trailing_serve_row(seen, C, 1)
    assert not np.isclose(r["t30_fn_delay"], r["t30_rt_delay"]) or not np.isclose(r["t30_rt_delay"], g_delay)
