"""
FlyRely BTS holdout backtest — implements backtest/PROTOCOL.md (pre-registered).

    python backtest/prepare.py --source marketing   # zips -> data/bts/parquet_v2
    python backtest/backtest.py     # writes backtest/results/*_v2data.* (never overwrites the registered results.json)

Models:
  committed  models/flight_delay_model.joblib, features built exactly as the API does
  booking    retrained on booking-time features (historical rates from prior months only)
  dayof      booking + same-aircraft previous-leg delay (proxy for what a day-of tracker knows)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import main  # noqa: E402  production feature builder + committed model paths

from flyrely_features.booking import (  # noqa: E402  shared with serving; re-exported for callers
    SMOOTHING, booking_features, days_to_holiday, smoothed_rate,
)
from paths import BTS_PARQUET as PARQUET  # noqa: E402  the one constant for the BTS data dir
import holdout as _holdout  # noqa: E402  the one holdout policy (backtest/holdout.py)
RESULTS = Path(__file__).resolve().parent / "results"
# The registered result (results.json, calibration.png, booking_model_candidate.joblib) was produced on the
# original data and must stay intact; re-runs on the all-carrier Marketing data write *_v2data files.
SUFFIX = "_v2data"
SEED = 42

# Rows each model sees, and the trailing 12 months its historical rates come from
PERIODS = {
    "fit":     {"rows": ("2025-01", "2025-06"), "rates": ("2024-01", "2024-12")},
    "val":     {"rows": ("2025-07", "2025-12"), "rates": ("2024-07", "2025-06")},
    "holdout": {"rows": ("2026-01", "2026-08"), "rates": ("2025-01", "2025-12")},
}
LOW, HIGH = 0.20, 0.30  # calibration v2 tier thresholds used in production


# =============================================================================
# Data
# =============================================================================

def load_months(start: str, end: str, holdout_dir: Path = None) -> pd.DataFrame:
    """Months start..end from the main folder. A protected holdout month (2026-09 on) is never read from there:
    it loads only from `holdout_dir` (data/holdout/YYYY-MM/bts, passed by the unlocked holdout stages), and only
    that one month's file, with no row outside the month."""
    months = pd.period_range(start, end, freq="M")
    frames = []
    for m in months:
        ym = f"{m.year}-{m.month:02d}"
        if not _holdout.is_locked_month(ym):
            frames.append(pd.read_parquet(PARQUET / f"{ym}.parquet"))
            continue
        if _holdout.unlocked_month(ym) != ym:       # the loader checks the unlock itself, not only its callers
            raise RuntimeError(f"{ym} is in the protected holdout and is not unlocked "
                               f"({_holdout.UNLOCK_VAR}={ym} with --holdout {ym} is required)")
        if holdout_dir is None or Path(holdout_dir).parent.name != ym:
            raise RuntimeError(f"{ym} is in the protected holdout (>= {_holdout.LOCKED_FROM}): "
                               "it loads only from data/holdout/YYYY-MM/bts by an unlocked holdout stage")
        part = pd.read_parquet(Path(holdout_dir) / f"{ym}.parquet")
        _holdout.refuse_rows_outside(part, ym)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    for c in ("Reporting_Airline", "Marketing_Airline_Network", "Origin", "Dest"):
        df[c] = df[c].astype(str)
    df = df[df["CRSDepTime"].notna()].copy()

    df["dep_hour"] = (df["CRSDepTime"].astype(int) // 100) % 24
    df["dow"] = df["FlightDate"].dt.dayofweek + 1  # ISO, matches API
    df["month"] = df["FlightDate"].dt.month
    df["ym"] = df["FlightDate"].dt.strftime("%Y-%m")
    df["cancelled"] = df["Cancelled"].fillna(0).astype(bool)
    df["diverted"] = df["Diverted"].fillna(0).astype(bool)
    df["delayed"] = np.where(df["cancelled"], np.nan, (df["DepDelay"] >= 15).astype(float))
    df["severe"] = np.where(df["cancelled"], np.nan, (df["DepDelay"] >= 120).astype(float))
    df["disrupted"] = ((df["DepDelay"] >= 15) | df["cancelled"] | df["diverted"]).astype(float)
    return df


# =============================================================================
# Features
# =============================================================================

def committed_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised copy of main.build_features (verified against it in check_parity)."""
    airline = df["Reporting_Airline"]  # operating carrier, as /predict resolves it via codeshare
    fallback = airline.map(main.AIRLINE_DELAY_RATES).fillna(0.20)
    origin_rate = df["Origin"].map(main.AIRPORT_DELAY_RATES).fillna(fallback)
    dest_rate = df["Dest"].map(main.AIRPORT_DELAY_RATES).fillna(fallback)
    known = df["Origin"].isin(main.AIRPORT_COORDS) & df["Dest"].isin(main.AIRPORT_COORDS)
    h = df["dep_hour"]
    dth = days_to_holiday(df["FlightDate"])
    distance = np.where(
        known,
        [main.estimate_distance(o, d) for o, d in zip(df["Origin"], df["Dest"])],
        250,
    )
    out = pd.DataFrame({
        "dep_hour": h,
        "day_of_week": df["dow"],
        "month": df["month"],
        "is_weekend": (df["dow"] >= 6).astype(int),
        "is_morning": ((h >= 5) & (h < 12)).astype(int),
        "is_afternoon": ((h >= 12) & (h < 18)).astype(int),
        "is_evening": ((h >= 18) & (h < 22)).astype(int),
        "is_night": ((h >= 22) | (h < 5)).astype(int),
        "season": df["month"].map(main.get_season),
        "is_holiday_period": (dth <= 3).astype(int),
        "days_to_holiday": dth,
        "origin_delay_rate": origin_rate,
        "dest_delay_rate": dest_rate,
        "route_delay_rate": (origin_rate + dest_rate) / 2,
        "airline_delay_rate": airline.map(main.AIRLINE_DELAY_RATES).fillna(0.21),
        "origin_is_hub": df["Origin"].isin(main.HUB_AIRPORTS).astype(int),
        "dest_is_hub": df["Dest"].isin(main.HUB_AIRPORTS).astype(int),
        "distance": distance,
    }, index=df.index)
    return out[main.model_service.feature_names]


def check_parity(df: pd.DataFrame, feats: pd.DataFrame, n: int = 2000) -> int:
    """Confirm the vectorised features equal main.build_features row by row."""
    sample = df.sample(min(n, len(df)), random_state=SEED)
    mismatches = 0
    for idx, r in sample.iterrows():
        dep = pd.Timestamp(r["FlightDate"]).to_pydatetime().replace(hour=int(r["dep_hour"]))
        ref = main.build_features(r["Origin"], r["Dest"], dep, r["Reporting_Airline"])
        row = feats.loc[idx]
        if any(not np.isclose(float(ref[k]), float(row[k])) for k in feats.columns):
            mismatches += 1
    return mismatches


def dayof_features(df: pd.DataFrame, booking: pd.DataFrame) -> pd.DataFrame:
    """Add what a day-of tracker knows hours before departure: the same aircraft's previous leg."""
    d = df[["Tail_Number", "FlightDate", "CRSDepTime", "CRSArrTime", "DepDelay", "cancelled"]].copy()
    d = d[d["Tail_Number"].notna()].sort_values(["Tail_Number", "FlightDate", "CRSDepTime"])
    same = (d["Tail_Number"] == d["Tail_Number"].shift()) & (d["FlightDate"] == d["FlightDate"].shift())
    to_min = lambda hhmm: (hhmm.astype(float) // 100) * 60 + hhmm.astype(float) % 100
    prev_delay = d["DepDelay"].shift().where(same)
    prev_cancel = d["cancelled"].shift().where(same)
    gap = (to_min(d["CRSDepTime"]) - to_min(d["CRSArrTime"].shift())).where(same)
    leg = d.groupby(["Tail_Number", "FlightDate"]).cumcount()

    out = booking.copy()
    out["prev_leg_dep_delay"] = prev_delay.reindex(df.index)
    out["prev_leg_cancelled"] = prev_cancel.astype(float).reindex(df.index)
    out["sched_turn_minutes"] = gap.reindex(df.index)
    out["leg_of_day"] = leg.reindex(df.index).astype(float)
    return out


# =============================================================================
# Metrics (exactly the PROTOCOL.md bar)
# =============================================================================

def ece_and_deciles(y: np.ndarray, p: np.ndarray) -> tuple[float, list[dict]]:
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, 10)
    table, ece = [], 0.0
    for i, b in enumerate(bins):
        pred, obs = float(p[b].mean()), float(y[b].mean())
        ece += len(b) / len(p) * abs(pred - obs)
        table.append({"decile": i + 1, "n": int(len(b)), "predicted": round(pred, 4), "observed": round(obs, 4)})
    return ece, table


def evaluate(df: pd.DataFrame, p: np.ndarray) -> dict:
    p = np.asarray(p, dtype=float)
    flown = ~df["cancelled"].to_numpy()
    y = df["delayed"].to_numpy()[flown]
    pf = p[flown]

    auc = roc_auc_score(y, pf)
    ece, deciles = ece_and_deciles(y, pf)
    max_gap = max(abs(d["predicted"] - d["observed"]) for d in deciles)

    tier = np.where(p < LOW, "low", np.where(p < HIGH, "medium", "high"))
    tiers = {}
    for t in ("low", "medium", "high"):
        m = (tier == t) & flown
        tiers[t] = {"share_of_flights": round(float((tier == t).mean()), 4),
                    "observed_delay_rate": round(float(df["delayed"].to_numpy()[m].mean()), 4) if m.any() else None}

    monthly, tiers_ordered_every_month = {}, True
    for ym, idx in df.groupby("ym").indices.items():
        mf = flown[idx]
        ym_y, ym_p, ym_t = df["delayed"].to_numpy()[idx][mf], p[idx][mf], tier[idx][mf]
        rates = {t: float(ym_y[ym_t == t].mean()) if (ym_t == t).any() else np.nan for t in ("low", "medium", "high")}
        ordered = rates["low"] < rates["medium"] < rates["high"]
        tiers_ordered_every_month &= bool(ordered)
        monthly[ym] = {"auc": round(roc_auc_score(ym_y, ym_p), 4), "tier_rates": {k: round(v, 4) for k, v in rates.items()},
                       "tiers_ordered": bool(ordered)}

    top20 = p >= np.quantile(p, 0.80)
    severe = df["severe"].to_numpy() == 1
    cancelled = df["cancelled"].to_numpy()
    severe_capture = float(severe[top20].sum() / severe.sum())
    cancel_capture = float(cancelled[top20].sum() / cancelled.sum())

    hi, lo = tiers["high"]["observed_delay_rate"], tiers["low"]["observed_delay_rate"]
    ratio = (hi / lo) if hi is not None and lo else None
    min_month_auc = min(m["auc"] for m in monthly.values())

    checks = {
        "1_auc_ge_0.70": auc >= 0.70,
        "2_ece_le_0.02_and_deciles_within_0.03": ece <= 0.02 and max_gap <= 0.03,
        "3_high_ge_2x_low_and_ordered_monthly": bool(ratio is not None and ratio >= 2 and tiers_ordered_every_month),
        "4_top20_captures_ge_35pct_severe": severe_capture >= 0.35,
        "5_top20_captures_ge_30pct_cancellations": cancel_capture >= 0.30,
        "6_monthly_auc_ge_0.65": min_month_auc >= 0.65,
    }
    return {
        "n_flights": int(len(df)), "n_flown": int(flown.sum()),
        "base_rates": {"delayed": round(float(y.mean()), 4), "severe": round(float(severe.sum() / flown.sum()), 4),
                       "cancelled": round(float(cancelled.mean()), 4), "disrupted": round(float(df["disrupted"].mean()), 4)},
        "auc_delayed": round(auc, 4),
        "auc_disrupted": round(roc_auc_score(df["disrupted"], p), 4),
        "auc_cancelled": round(roc_auc_score(cancelled, p), 4),
        "brier_delayed": round(brier_score_loss(y, pf), 4),
        "ece": round(ece, 4), "max_decile_gap": round(max_gap, 4), "deciles": deciles,
        "tiers": tiers, "high_low_ratio": round(ratio, 3) if ratio else None,
        "tiers_ordered_every_month": tiers_ordered_every_month,
        "top20_severe_capture": round(severe_capture, 4), "top20_cancel_capture": round(cancel_capture, 4),
        "min_monthly_auc": min_month_auc, "monthly": monthly,
        "checks": checks, "all_pass": all(checks.values()),
    }


# =============================================================================
# Run
# =============================================================================

def fit_calibrated(X_fit, y_fit, X_val, y_val):
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=SEED,
    )
    model.fit(X_fit, y_fit)
    iso = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(X_val)[:, 1], y_val)
    return model, iso


def main_run():
    RESULTS.mkdir(exist_ok=True)
    everything = load_months("2024-01", "2026-08")
    window = lambda a, b: everything[(everything["ym"] >= a) & (everything["ym"] <= b)]
    data = {k: window(*v["rows"]) for k, v in PERIODS.items()}
    hist = {k: window(*v["rates"]) for k, v in PERIODS.items()}
    hold = data["holdout"]
    for k, d in data.items():
        print(f"{k}: {len(d):,} flights {d['ym'].min()}..{d['ym'].max()}")

    results = {"protocol_commit": "f6678e2", "periods": PERIODS}

    # 1. Committed model, exactly as production builds features
    cf = committed_features(hold)
    results["committed_parity_mismatches_of_2000"] = check_parity(hold, cf)
    p_committed = main.model_service.model.predict_proba(cf)[:, 1]
    results["committed"] = evaluate(hold, p_committed)
    print("committed AUC", results["committed"]["auc_delayed"])

    # 2. Booking-time model
    Xb = {k: booking_features(data[k], hist[k]) for k in data}
    fit_mask = ~data["fit"]["cancelled"].to_numpy()
    val_mask = ~data["val"]["cancelled"].to_numpy()
    yb_fit, yb_val = data["fit"]["delayed"].to_numpy()[fit_mask], data["val"]["delayed"].to_numpy()[val_mask]
    m_b, iso_b = fit_calibrated(Xb["fit"][fit_mask], yb_fit, Xb["val"][val_mask], yb_val)
    p_booking = iso_b.predict(m_b.predict_proba(Xb["holdout"])[:, 1])
    results["booking"] = evaluate(hold, p_booking)
    results["booking"]["features"] = list(Xb["fit"].columns)
    print("booking AUC", results["booking"]["auc_delayed"])

    # 3. Day-of reference (previous-leg delay)
    Xd = {k: dayof_features(data[k], Xb[k]) for k in data}
    m_d, iso_d = fit_calibrated(Xd["fit"][fit_mask], yb_fit, Xd["val"][val_mask], yb_val)
    p_dayof = iso_d.predict(m_d.predict_proba(Xd["holdout"])[:, 1])
    results["dayof"] = evaluate(hold, p_dayof)
    results["dayof"]["features"] = list(Xd["fit"].columns)
    print("dayof AUC", results["dayof"]["auc_delayed"])

    # Failure analysis: where the booking model is worst (holdout, flown flights)
    flown = ~hold["cancelled"].to_numpy()
    fa = hold.loc[flown, ["Reporting_Airline", "Origin", "dep_hour", "month"]].copy()
    fa["y"], fa["p"] = hold["delayed"].to_numpy()[flown], p_booking[flown]
    fa["abs_err"] = (fa["y"] - fa["p"]).abs()
    def worst(col, min_n=5000):
        g = fa.groupby(col).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
        g = g[g["n"] >= min_n]
        g["gap"] = g["observed"] - g["predicted"]
        return g.reindex(g["gap"].abs().sort_values(ascending=False).index).head(8).round(4).reset_index().to_dict("records")
    results["failure_analysis"] = {c: worst(c) for c in ("Reporting_Airline", "Origin", "dep_hour", "month")}

    (RESULTS / f"results{SUFFIX}.json").write_text(json.dumps(results, indent=2, default=float))
    joblib.dump({"model": m_b, "calibrator": iso_b, "features": list(Xb["fit"].columns)},
                RESULTS / f"booking_model_candidate{SUFFIX}.joblib")
    plot_calibration(results)
    print(json.dumps({k: results[k]["checks"] for k in ("committed", "booking", "dayof")}, indent=2))


def plot_calibration(results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 0.7], [0, 0.7], color="#999", lw=1, ls="--", label="perfect")
    for name, color in (("committed", "#c0392b"), ("booking", "#2471a3"), ("dayof", "#7d3c98")):
        d = results[name]["deciles"]
        ax.plot([x["predicted"] for x in d], [x["observed"] for x in d], "o-", color=color,
                label=f"{name} (AUC {results[name]['auc_delayed']:.3f})")
    ax.set_xlabel("Predicted delay probability (decile mean)")
    ax.set_ylabel("Observed delay rate")
    ax.set_title("Calibration on 2026 holdout")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"calibration{SUFFIX}.png", dpi=150)


if __name__ == "__main__":
    main_run()
