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

ROOT = bt.ROOT
WEATHER = ROOT / "data" / "weather"
RESULTS = bt.RESULTS
SEED = bt.SEED

HORIZONS = {"7d": 7 * 1440, "1d": 1440, "3h": 180, "1h": 60}  # minutes before departure
SAME_DAY = {"3h", "1h"}
MIN_TURN = 35            # minutes an aircraft needs on the ground between legs
SHIFT = 2000             # keeps every clock time positive (Guam is UTC+10)
T = 2_000_000            # key stride per group; larger than any time in the data
DAY0 = pd.Timestamp("2024-01-01")
WEATHER_VARS = ["precipitation", "snowfall", "wind_gusts_10m", "cape", "weather_code"]

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


def airport_offsets(o, d, ym, dep, arr, elapsed, n_airports, anchor) -> np.ndarray:
    """Minutes each airport's local clock is ahead of `anchor` (ATL), per month.

    For every route, arr - dep - elapsed = offset(dest) - offset(origin). Take the
    most common value per route and month, then walk the route graph from ATL.
    """
    diff = (arr - dep - elapsed) % 1440
    diff = np.round(np.where(diff > 720, diff - 1440, diff) / 60) * 60
    e = pd.DataFrame({"ym": ym, "o": o, "d": d, "diff": diff}).dropna()
    e = (e.groupby(["ym", "o", "d", "diff"]).size().rename("n").reset_index()
         .sort_values("n", ascending=False).drop_duplicates(["ym", "o", "d"]))
    n_months = int(ym.max()) + 1
    off = np.full((n_months, n_airports), np.nan)
    for m, g in e.groupby("ym"):
        known = {anchor: 0.0}
        edges = list(zip(g["o"], g["d"], g["diff"]))
        changed = True
        while changed:
            changed = False
            for a, b, w in edges:
                if a in known and b not in known:
                    known[b] = known[a] + w
                    changed = True
                elif b in known and a not in known:
                    known[a] = known[b] - w
                    changed = True
        for a, v in known.items():
            off[int(m), a] = v
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

def load() -> tuple[pd.DataFrame, dict]:
    df = bt.load_months("2024-01", "2025-08" if QUICK else "2026-08").reset_index(drop=True)
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
    arr_local = hhmm_to_min(df["CRSArrTime"])
    elapsed = df["CRSElapsedTime"].astype("float64").to_numpy()

    off = airport_offsets(A["o_id"], A["d_id"], A["ym_id"], A["dep_local"], arr_local, elapsed,
                          len(airports), airports.get_loc("ATL"))
    off_o = off[A["ym_id"], A["o_id"]]
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
# Counting events per group up to a time, with one searchsorted
# =============================================================================

class Counter:
    """Running sums of event values per group, queryable at any time cutoff."""

    def __init__(self, gid: np.ndarray, t: np.ndarray, values: dict[str, np.ndarray]):
        m = (gid >= 0) & ~np.isnan(t)
        key = gid[m] * T + np.floor(t[m]).astype(np.int64)
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.cum = {k: np.concatenate([[0.0], np.cumsum(np.asarray(v, dtype="float64")[m][order])])
                    for k, v in values.items()}
        self.cum["n"] = np.arange(len(self.key) + 1, dtype="float64")

    def upto(self, gid: np.ndarray, t: np.ndarray, name: str = "n") -> np.ndarray:
        bad = (gid < 0) | np.isnan(t)
        g = np.where(bad, 0, gid)
        q = g * T + np.floor(np.nan_to_num(t)).astype(np.int64)
        hi = np.searchsorted(self.key, q, side="right")
        lo = np.searchsorted(self.key, g * T, side="left")
        out = self.cum[name][hi] - self.cum[name][lo]
        return np.where(bad, np.nan, out)

    def window(self, gid, lo_excl, hi_incl, name="n"):
        return self.upto(gid, hi_incl, name) - self.upto(gid, lo_excl, name)


def build_counters(A: dict) -> dict:
    flown = ~A["cancelled"]
    late = np.maximum(np.nan_to_num(A["dep_delay"]), 0)
    # A flight with no recorded delay is treated as gone on time, so it never sits "pending" forever
    done_t = np.where(A["cancelled"], np.nan, np.fmax(A["s_dep"], A["a_dep"]))
    C = {}
    for name, gid in (("o", A["o_id"]), ("al", A["al_id"])):
        C[f"dep_{name}"] = Counter(gid, A["a_dep"], {"d15": A["d15"], "late": late})
        C[f"sched_{name}"] = Counter(gid, A["s_dep"], {"canc": A["cancelled"], "s": A["s_dep"],
                                                       "cs": A["s_dep"] * A["cancelled"]})
        C[f"done_{name}"] = Counter(gid, done_t, {"s": A["s_dep"]})
    day = A["day"].astype("float64")
    vals = {"flown": flown, "d15": A["d15"], "canc": A["cancelled"]}
    for name in ("o", "d", "al", "rt", "fn"):
        C[f"day_{name}"] = Counter(A[f"{name}_id"], day, vals)
    return C


# =============================================================================
# Features known at the cutoff
# =============================================================================

def cutoff_day(A, rows, h):
    return A["day"][rows] + np.floor((A["dep_local"][rows] - HORIZONS[h]) / 1440)


def recent_features(A, C, rows, h) -> dict:
    """Trailing 7-day performance, window ending the day before the cutoff date."""
    end = cutoff_day(A, rows, h) - 1
    out = {}
    for name in ("o", "d", "al", "rt", "fn"):
        c, g = C[f"day_{name}"], A[f"{name}_id"][rows]
        n, flown = c.window(g, end - 7, end), c.window(g, end - 7, end, "flown")
        d15, canc = c.window(g, end - 7, end, "d15"), c.window(g, end - 7, end, "canc")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"r7_{name}_delay_rate"] = np.where(flown > 0, d15 / flown, np.nan)
            out[f"r7_{name}_cancel_rate"] = np.where(n > 0, canc / n, np.nan)
        out[f"r7_{name}_flights"] = n
    return out


def flight_history(A, C, rows, end_day) -> np.ndarray:
    """Flighty-style on-time history: flight number's 90-day delay rate, shrunk toward its airline."""
    fn, al = C["day_fn"], C["day_al"]
    g_fn, g_al = A["fn_id"][rows], A["al_id"][rows]
    al_flown, al_d15 = al.window(g_al, end_day - 90, end_day, "flown"), al.window(g_al, end_day - 90, end_day, "d15")
    prior = np.where(al_flown > 0, al_d15 / np.maximum(al_flown, 1), 0.2)
    flown, d15 = fn.window(g_fn, end_day - 90, end_day, "flown"), fn.window(g_fn, end_day - 90, end_day, "d15")
    return (d15 + 10 * prior) / (flown + 10)


def network_features(A, C, rows, c) -> dict:
    """Same-day state at cutoff c: origin, destination and the airline's whole system."""
    out = {}
    for prefix, counter, gq in (("o", "o", A["o_id"]), ("d", "o", A["d_id"]), ("al", "al", A["al_id"])):
        g = gq[rows]
        dep, sch, done = C[f"dep_{counter}"], C[f"sched_{counter}"], C[f"done_{counter}"]
        for w in (120, 480):
            n = dep.window(g, c - w, c)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{prefix}_{w}m_delayed_share"] = np.where(n > 0, dep.window(g, c - w, c, "d15") / n, np.nan)
                out[f"{prefix}_{w}m_mean_late"] = np.where(n > 0, dep.window(g, c - w, c, "late") / n, np.nan)
            out[f"{prefix}_{w}m_departed"] = n
        out[f"{prefix}_480m_cancelled"] = sch.window(g, c - 480, c, "canc")
        pending = sch.upto(g, c) - done.upto(g, c) - sch.upto(g, c, "canc")
        s_sum = sch.upto(g, c, "s") - done.upto(g, c, "s") - sch.upto(g, c, "cs")
        out[f"{prefix}_pending"] = pending
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{prefix}_pending_mean_wait"] = np.where(pending > 0, c - s_sum / np.maximum(pending, 1), 0.0)
    return out


class Rotation:
    """Every leg flown by each aircraft, in schedule order."""

    def __init__(self, A):
        m = (A["tail_id"] >= 0) & ~np.isnan(A["s_dep"])
        idx = np.flatnonzero(m)
        key = A["tail_id"][idx] * T + np.floor(A["s_dep"][idx]).astype(np.int64)
        order = np.argsort(key, kind="stable")
        self.rows = idx[order]
        self.key = key[order]
        self.pos = np.full(len(A["s_dep"]), -1, dtype=np.int64)
        self.pos[self.rows] = np.arange(len(self.rows))
        self.cum_elapsed = np.cumsum(A["elapsed"][self.rows])


def rotation_features(A, R: Rotation, rows, c) -> dict:
    """Where this flight's aircraft is at cutoff c: the latest leg already scheduled to have left."""
    n = len(rows)
    self_pos = R.pos[rows]
    tail = A["tail_id"][rows]
    has = self_pos >= 0
    q = np.where(has, tail * T + np.floor(np.nan_to_num(c)).astype(np.int64), 0)
    j = np.searchsorted(R.key, q, side="right") - 1
    jj = np.clip(j, 0, len(R.rows) - 1)
    leg = R.rows[jj]
    valid = has & (j >= 0) & (R.key[jj] // T == tail) & (A["s_dep"][leg] > c - 18 * 60) & (j < self_pos)

    s_leg, a_leg = A["s_dep"][leg], A["a_dep"][leg]
    departed = a_leg <= c
    known = np.where(departed, np.nan_to_num(A["dep_delay"][leg]), np.maximum(c - s_leg, 0))
    leg_cancelled = A["cancelled"][leg]
    known = np.where(leg_cancelled, np.nan, known)

    between = np.where(valid, self_pos - j - 1, 0)
    sp = np.clip(self_pos - 1, 0, None)
    block = np.where(between > 0, R.cum_elapsed[sp] - R.cum_elapsed[jj], 0.0)
    slack = A["s_dep"][rows] - A["s_arr"][leg] - block - (between + 1) * MIN_TURN

    prev = R.rows[np.clip(self_pos - 1, 0, len(R.rows) - 1)]
    prev_ok = has & (self_pos > 0) & (A["tail_id"][prev] == tail) & (A["s_dep"][prev] > A["s_dep"][rows] - 18 * 60)

    nan = lambda x: np.where(valid, x, np.nan)
    return {
        "rot_has_leg": valid.astype(float),
        "rot_known_delay": nan(known),
        "rot_leg_pending": nan((~departed & ~leg_cancelled).astype(float)),
        "rot_leg_cancelled": nan(leg_cancelled.astype(float)),
        "rot_legs_between": nan(between.astype(float)),
        "rot_slack": nan(slack),
        "rot_projected_delay": nan(np.maximum(known - slack, 0)),
        "rot_minutes_since_leg_sched": nan(c - s_leg),
        "rot_sched_turn": np.where(prev_ok, A["s_dep"][rows] - A["s_arr"][prev], np.nan),
    }


class Weather:
    """Archived forecasts per airport and UTC hour (Open-Meteo Previous Runs)."""

    def __init__(self, airports: pd.Index):
        files = sorted(WEATHER.glob("*.parquet"))
        self.ok = bool(files)
        if not self.ok:
            return
        w = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        self.n_airports = w["airport"].nunique()
        ap = airports.get_indexer(w["airport"])
        hour = ((w["utc_hour"] - DAY0) / pd.Timedelta(hours=1)).astype(np.int64).to_numpy()
        keep = ap >= 0
        self.index = pd.Index(ap[keep] * T + hour[keep])
        self.values = w.loc[keep].reset_index(drop=True)

    def features(self, A, rows, lead: int) -> dict:
        out = {}
        for side, ap, t in (("o", A["o_id"][rows], A["s_dep"][rows]), ("d", A["d_id"][rows], A["s_arr"][rows])):
            utc = t - SHIFT + A["utc_shift"][rows]
            hour = np.floor(np.nan_to_num(utc) / 60).astype(np.int64)
            i = self.index.get_indexer(ap * T + hour)
            for v in WEATHER_VARS:
                col = self.values[f"{v}_previous_day{lead}"].to_numpy(dtype="float64")
                out[f"wx_{side}_{v}"] = np.where(i >= 0, col[np.clip(i, 0, None)], np.nan)
        return out


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
