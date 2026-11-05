"""
FlyRely run-up backtest — implements backtest/PROTOCOL_RUNUP.md (pre-registered).

    python backtest/prepare.py --source marketing   # zips -> data/bts/parquet_v2 (read via backtest.PARQUET)
    python backtest/fetch_weather.py    # optional: enables weather variant W
    python backtest/runup.py            # writes backtest/results/runup.json
    python backtest/runup.py --quick    # debug run on validation months only (never touches holdout)

For each horizon (T-7d, T-1d, T-3h, T-1h) it trains a model on what is known at
the cutoff `c = scheduled departure - h`, and compares it with a Flighty proxy
for the same moment.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import backtest as bt
from flyrely_features._common import (  # noqa: F401  re-exported: tests and callers use runup.<name>
    DAY0, HORIZONS, MIN_TURN, SAME_DAY, SHIFT, T, WEATHER_VARS, Counter, build_counters, cutoff_day,
)
from flyrely_features.network import network_features  # noqa: F401
from flyrely_features.recent import recent_features
from flyrely_features.rotation import Rotation, rotation_features
from flyrely_features.weather import WEATHER, Weather  # noqa: F401

ROOT = bt.ROOT
RESULTS = bt.RESULTS
SEED = bt.SEED

QUICK = "--quick" in sys.argv
PERIODS = {
    "fit": bt.PERIODS["fit"], "val": bt.PERIODS["val"], "holdout": bt.PERIODS["holdout"],
} if not QUICK else {  # validation months stand in for the holdout, so debugging never peeks at it
    "fit": {"rows": ("2025-01", "2025-02"), "rates": ("2024-01", "2024-12")},
    "val": {"rows": ("2025-07", "2025-07"), "rates": ("2024-07", "2025-06")},
    "holdout": {"rows": ("2025-08", "2025-08"), "rates": ("2024-07", "2025-06")},
}


# =============================================================================
# Clock: one reference time zone for every airport, derived from BTS itself
# =============================================================================

def hhmm_to_min(s: pd.Series) -> np.ndarray:
    v = s.astype("float64").to_numpy()
    return (v // 100) * 60 + v % 100


AIRPORT_TZ_CSV = ROOT / "models" / "airports_ref.csv"


def airport_zones(airports: pd.Index, csv=None) -> list:
    """IANA time zone per airport from models/airports_ref.csv (None where the airport is not listed)."""
    tz = pd.read_csv(csv or AIRPORT_TZ_CSV, dtype=str).set_index("iata")["tz"]
    return [tz.get(a) for a in airports]


def utc_offset_minutes(zone: str, n_days: int) -> np.ndarray:
    """UTC offset (minutes ahead of UTC) of `zone` at local noon of each day since DAY0."""
    from zoneinfo import ZoneInfo
    z = ZoneInfo(zone)
    return np.array([(DAY0 + pd.Timedelta(days=i, hours=12)).to_pydatetime().replace(tzinfo=z).utcoffset().total_seconds() / 60
                     for i in range(n_days)])


def airport_offsets(airports: pd.Index, n_days: int, anchor: str = "ATL", zones: list = None) -> np.ndarray:
    """Minutes each airport's local clock is ahead of `anchor` (ATL), per day: shape (n_days, n_airports).

    From the tz database, so airports without daylight saving time (PHX, HNL, SJU, ...) are right on the clock-change
    days (B5.1 fix (d); the earlier route-graph derivation was one month wide). Offsets are taken at local noon, so
    flights in the few hours before the change on the two change days use the post-change offset.
    NaN where the airport has no known zone."""
    zones = zones if zones is not None else airport_zones(airports)
    ref = utc_offset_minutes(zones[airports.get_loc(anchor)], n_days)
    cache = {}
    off = np.full((n_days, len(airports)), np.nan)
    for j, z in enumerate(zones):
        if z is None:
            continue
        if z not in cache:
            cache[z] = utc_offset_minutes(z, n_days)
        off[:, j] = cache[z] - ref
    return off


def eastern_to_utc_minutes(days: np.ndarray) -> np.ndarray:
    """Per day index, minutes to add to US Eastern local time to get UTC."""
    out = np.empty(days.max() + 1)
    for i in range(len(out)):
        noon = (DAY0 + pd.Timedelta(days=i, hours=12)).tz_localize("America/New_York")
        out[i] = -noon.utcoffset().total_seconds() / 60
    return out


# =============================================================================
# Data
# =============================================================================

def load(end: str = None, holdout_dir=None) -> tuple[pd.DataFrame, dict]:
    """Months 2024-01..`end` (default: 2026-08, the last non-holdout month). `end` and `holdout_dir` are set only
    by the unlocked holdout stages of experiments_b5.py; the defaults are exactly the B5.1 behaviour."""
    df = bt.load_months("2024-01", end or ("2025-08" if QUICK else "2026-08"), holdout_dir).reset_index(drop=True)
    airports = pd.Index(sorted(set(df["Origin"]) | set(df["Dest"])))
    airlines = pd.Index(sorted(df["Reporting_Airline"].unique()))
    yms = pd.Index(sorted(df["ym"].unique()))

    A = {"airports": airports}
    A["o_id"] = airports.get_indexer(df["Origin"]).astype(np.int64)
    A["d_id"] = airports.get_indexer(df["Dest"]).astype(np.int64)
    A["al_id"] = airlines.get_indexer(df["Reporting_Airline"]).astype(np.int64)
    A["rt_id"] = A["o_id"] * len(airports) + A["d_id"]
    fn = A["al_id"] * 100_000 + df["Flight_Number_Reporting_Airline"].fillna(0).astype(np.int64).to_numpy()
    A["fn_id"] = pd.factorize(fn)[0].astype(np.int64)
    A["tail_id"] = pd.factorize(df["Tail_Number"], use_na_sentinel=True)[0].astype(np.int64)
    A["ym_id"] = yms.get_indexer(df["ym"])
    A["day"] = (df["FlightDate"] - DAY0).dt.days.to_numpy().astype(np.int64)
    A["dep_local"] = hhmm_to_min(df["CRSDepTime"])
    elapsed = df["CRSElapsedTime"].astype("float64").to_numpy()

    off = airport_offsets(airports, int(A["day"].max()) + 1)
    off_o = off[A["day"], A["o_id"]]
    A["s_dep"] = A["day"] * 1440 + A["dep_local"] - off_o + SHIFT  # Eastern-reference minutes
    A["s_arr"] = A["s_dep"] + elapsed
    A["elapsed"] = np.nan_to_num(elapsed)

    dep_delay = df["DepDelay"].astype("float64").to_numpy()
    A["cancelled"] = df["cancelled"].to_numpy()
    A["dep_delay"] = dep_delay
    A["a_dep"] = np.where(A["cancelled"], np.nan, A["s_dep"] + np.nan_to_num(dep_delay))
    A["d15"] = (np.nan_to_num(dep_delay) >= 15) & ~A["cancelled"]
    A["utc_shift"] = eastern_to_utc_minutes(A["day"])[A["day"]]
    print(f"clock: {np.isnan(off_o).mean():.4%} of flights at airports with no derived offset")
    return df, A


# =============================================================================
# Features known at the cutoff (recent/network/rotation/weather live in flyrely_features)
# =============================================================================

def flight_history(A, C, rows, end_day) -> np.ndarray:
    """Flighty-style on-time history: flight number's 90-day delay rate, shrunk toward its airline."""
    fn, al = C["day_fn"], C["day_al"]
    g_fn, g_al = A["fn_id"][rows], A["al_id"][rows]
    al_flown, al_d15 = al.window(g_al, end_day - 90, end_day, "flown"), al.window(g_al, end_day - 90, end_day, "d15")
    prior = np.where(al_flown > 0, al_d15 / np.maximum(al_flown, 1), 0.2)
    flown, d15 = fn.window(g_fn, end_day - 90, end_day, "flown"), fn.window(g_fn, end_day - 90, end_day, "d15")
    return (d15 + 10 * prior) / (flown + 10)


def horizon_features(A, C, R, W, rows, h, booking_X, weather: bool) -> pd.DataFrame:
    c = A["s_dep"][rows] - HORIZONS[h]
    f = recent_features(A, C, rows, h)
    if h in SAME_DAY:
        f.update(network_features(A, C, rows, c))
        f.update(rotation_features(A, R, rows, c))
    if weather:
        f.update(W.features(A, rows, 7 if h == "7d" else 1))
    extra = pd.DataFrame(f, index=booking_X.index).astype("float32")
    return pd.concat([booking_X, extra], axis=1)


def proxy_inputs(A, C, R, rows, h) -> np.ndarray:
    """Inputs of the Flighty proxy at horizon h (see PROTOCOL_RUNUP.md)."""
    hist = flight_history(A, C, rows, cutoff_day(A, rows, h) - 1)
    if h not in SAME_DAY:
        return hist[:, None]
    rot = rotation_features(A, R, rows, A["s_dep"][rows] - HORIZONS[h])
    proj = np.nan_to_num(rot["rot_projected_delay"])
    logit = np.log(np.clip(hist, 1e-3, 1 - 1e-3) / (1 - np.clip(hist, 1e-3, 1 - 1e-3)))
    return np.column_stack([logit, np.minimum(proj, 300), np.log1p(np.minimum(proj, 300)),
                            rot["rot_has_leg"], np.nan_to_num(rot["rot_leg_pending"])])


# =============================================================================
# Scoring (PROTOCOL_RUNUP.md)
# =============================================================================

def score(frame: pd.DataFrame, p: np.ndarray) -> dict:
    flown = ~frame["cancelled"].to_numpy()
    y, pf = frame["delayed"].to_numpy()[flown], p[flown]
    ece, deciles = bt.ece_and_deciles(y, pf)
    monthly = {ym: round(roc_auc_score(frame["delayed"].to_numpy()[i][flown[i]], p[i][flown[i]]), 4)
               for ym, i in frame.groupby("ym").indices.items()}
    return {"auc_delayed": round(roc_auc_score(y, pf), 4), "brier_delayed": round(brier_score_loss(y, pf), 5),
            "ece": round(ece, 4), "max_decile_gap": round(max(abs(d["predicted"] - d["observed"]) for d in deciles), 4),
            "auc_disrupted": round(roc_auc_score(frame["disrupted"], p), 4), "deciles": deciles, "monthly_auc": monthly}


def brier_diff(frame: pd.DataFrame, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 1000) -> dict:
    """Brier(a) - Brier(b) on flown flights, with a 95% day-block bootstrap interval."""
    flown = ~frame["cancelled"].to_numpy()
    y = frame["delayed"].to_numpy()[flown]
    d = pd.DataFrame({"day": frame["FlightDate"].to_numpy()[flown],
                      "a": (p_a[flown] - y) ** 2, "b": (p_b[flown] - y) ** 2})
    d = d.groupby("day").agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    idx = np.random.default_rng(SEED).integers(0, len(d), size=(n_boot, len(d)))
    boot = (d["a"].to_numpy()[idx].sum(1) - d["b"].to_numpy()[idx].sum(1)) / d["n"].to_numpy()[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": round(float((d["a"].sum() - d["b"].sum()) / d["n"].sum()), 6),
            "ci95": [round(float(lo), 6), round(float(hi), 6)]}


def isotonic(x_val, y_val):
    return IsotonicRegression(out_of_bounds="clip").fit(x_val, y_val)


# =============================================================================
# Run
# =============================================================================

def main_run():
    RESULTS.mkdir(exist_ok=True)
    df, A = load()
    W = Weather(A["airports"])
    C, R = build_counters(A), Rotation(A)
    print("counters built; weather airports:", W.n_airports if W.ok else 0)

    within = lambda a, b: np.flatnonzero(((df["ym"] >= a) & (df["ym"] <= b)).to_numpy())
    rows = {k: within(*v["rows"]) for k, v in PERIODS.items()}
    frame = {k: df.iloc[r] for k, r in rows.items()}
    booking_X = {}
    for k, v in PERIODS.items():
        hist = df.iloc[within(*v["rates"])]
        booking_X[k] = bt.booking_features(frame[k], hist).astype("float32")
    flown = {k: ~frame[k]["cancelled"].to_numpy() for k in rows}
    y = {k: frame[k]["delayed"].to_numpy()[flown[k]] for k in rows}
    hold = frame["holdout"]

    results = {"protocol": "backtest/PROTOCOL_RUNUP.md", "quick_debug_run": QUICK, "periods": PERIODS,
               "weather_airports": W.n_airports if W.ok else 0}

    # Booking time: FlyRely booking model (PROTOCOL.md model 2) vs flight-history proxy
    m_b, iso_b = bt.fit_calibrated(booking_X["fit"][flown["fit"]], y["fit"],
                                   booking_X["val"][flown["val"]], y["val"])
    p_book = {k: iso_b.predict(m_b.predict_proba(booking_X[k])[:, 1]) for k in ("val", "holdout")}
    fh = {k: flight_history(A, C, rows[k], A["day"][rows[k]] - 31) for k in ("val", "holdout")}
    iso_fh = isotonic(fh["val"][flown["val"]], y["val"])
    p_fh = iso_fh.predict(fh["holdout"])
    results["booking"] = {"flyrely": score(hold, p_book["holdout"]), "flighty_proxy": score(hold, p_fh),
                          "brier_flyrely_minus_proxy": brier_diff(hold, p_book["holdout"], p_fh)}
    print("booking", results["booking"]["flyrely"]["auc_delayed"], "proxy", results["booking"]["flighty_proxy"]["auc_delayed"])

    for h in HORIZONS:
        # Flighty proxy for this horizon
        P = {k: proxy_inputs(A, C, R, rows[k], h) for k in rows}
        if h in SAME_DAY:
            lr = LogisticRegression(max_iter=1000).fit(P["fit"][flown["fit"]], y["fit"])
            raw = {k: lr.predict_proba(P[k])[:, 1] for k in ("val", "holdout")}
        else:
            raw = {k: P[k][:, 0] for k in ("val", "holdout")}
        p_proxy = isotonic(raw["val"][flown["val"]], y["val"]).predict(raw["holdout"])

        # FlyRely horizon model, with and without weather; the choice is made on validation
        variants = [False, True] if W.ok else [False]
        fitted = {}
        for wx in variants:
            X = {k: horizon_features(A, C, R, W, rows[k], h, booking_X[k], wx) for k in ("fit", "val")}
            m, iso = bt.fit_calibrated(X["fit"][flown["fit"]], y["fit"], X["val"][flown["val"]], y["val"])
            val_brier = brier_score_loss(y["val"], iso.predict(m.predict_proba(X["val"][flown["val"]])[:, 1]))
            fitted[wx] = (m, iso, val_brier, list(X["fit"].columns))
            del X
        chosen = min(variants, key=lambda wx: fitted[wx][2])
        m, iso, _, cols = fitted[chosen]
        X_hold = horizon_features(A, C, R, W, rows["holdout"], h, booking_X["holdout"], chosen)
        p = iso.predict(m.predict_proba(X_hold)[:, 1])
        del X_hold

        r = {"weather_used": chosen, "validation_brier": {str(k): round(v[2], 6) for k, v in fitted.items()},
             "features": cols, "flyrely": score(hold, p), "flighty_proxy": score(hold, p_proxy),
             "brier_flyrely_minus_proxy": brier_diff(hold, p, p_proxy),
             "brier_flyrely_minus_booking": brier_diff(hold, p, p_book["holdout"])}
        r["checks"] = {
            "1_brier_beats_proxy_ci_below_zero": r["brier_flyrely_minus_proxy"]["ci95"][1] < 0,
            "2_auc_beats_proxy": r["flyrely"]["auc_delayed"] > r["flighty_proxy"]["auc_delayed"],
            "3_ece_le_0.02": r["flyrely"]["ece"] <= 0.02,
            "4_brier_beats_booking": r["brier_flyrely_minus_booking"]["diff"] < 0,
        }
        r["beats_flighty_proxy"] = all(r["checks"].values())
        results[h] = r
        joblib.dump({"model": m, "calibrator": iso, "features": cols, "horizon": h},
                    RESULTS / f"runup_{h}{'_quick' if QUICK else ''}.joblib")
        print(h, "flyrely", r["flyrely"]["auc_delayed"], r["flyrely"]["brier_delayed"],
              "proxy", r["flighty_proxy"]["auc_delayed"], r["flighty_proxy"]["brier_delayed"], r["checks"])

    aucs = [results[h]["flyrely"]["auc_delayed"] for h in HORIZONS]
    results["auc_non_decreasing_toward_departure"] = all(b >= a - 0.002 for a, b in zip(aucs, aucs[1:]))
    out = RESULTS / ("runup_quick.json" if QUICK else "runup.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main_run()
