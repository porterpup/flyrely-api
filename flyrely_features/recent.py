"""Trailing 7-day recent performance (delay/cancel rate and volume) for origin, destination, airline, route, flight number."""

import numpy as np

from ._common import T, Counter, cutoff_day


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


def serve_row(query: dict, C: dict, h: str) -> dict:
    """One flight. `query` holds 1-element arrays: day, dep_local and o_id/d_id/al_id/rt_id/fn_id.
    `C` are the counters built from history; same code path as the vectorised call."""
    out = recent_features(query, C, np.array([0]), h)
    return {k: float(v[0]) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Booking-time trailing rates with a data gap (experiment B1)
# ---------------------------------------------------------------------------
TRAIL_GROUPS = ("o", "d", "al", "nw", "rt", "oh", "fn")  # origin, dest, operating carrier, network, route, flight no., origin x hour
TRAIL_WINDOWS = (30, 90)
TRAIL_K = 50  # pseudo-flights pulling a sparse group toward its parent rate
TRAIL_PARENT = {"o": "g", "d": "g", "al": "g", "nw": "g", "rt": "od", "oh": "o", "fn": "rt"}
LONGRUN = {"al": ("rate_Reporting_Airline_delayed", "rate_Reporting_Airline_c"),
           "o": ("rate_Origin_delayed", "rate_Origin_c")}
_STATS = ("cnt", "flown", "d15", "canc")


def daily_counters(A) -> dict:
    """Per-group running sums over days. A needs: day, cancelled, d15 and <group>_id for each of TRAIL_GROUPS and g."""
    vals = {"cnt": np.ones(len(A["day"])), "flown": ~A["cancelled"], "d15": A["d15"], "canc": A["cancelled"]}
    C = {}
    for name in TRAIL_GROUPS + ("g",):
        gid = np.asarray(A[f"{name}_id"]).astype(np.int64)
        m = gid >= 0
        uniq, inv = np.unique(gid[m] * T + np.asarray(A["day"])[m].astype(np.int64), return_inverse=True)
        sums = {k: np.bincount(inv, weights=np.asarray(v, dtype="float64")[m], minlength=len(uniq)) for k, v in vals.items()}
        C[f"td_{name}"] = Counter(uniq // T, (uniq % T).astype("float64"), sums)
    return C


def _window(counter, g, lo, hi) -> dict:
    """Sums over days in (lo, hi] for group ids g (negative id -> NaN)."""
    bad = g < 0
    base = np.where(bad, 0, g).astype(np.int64) * T
    clip = lambda x: np.clip(np.floor(x), -1, T - 1).astype(np.int64)
    i_hi = np.searchsorted(counter.key, base + clip(hi), side="right")
    i_lo = np.searchsorted(counter.key, base + clip(lo), side="right")
    return {k: np.where(bad, np.nan, counter.cum[k][i_hi] - counter.cum[k][i_lo]) for k in _STATS}


def trail_names() -> list:
    return [f"t{w}_{n}_{k}" for w in TRAIL_WINDOWS for n in TRAIL_GROUPS for k in ("delay", "cancel")]


def trailing_features(A, C, rows, gap, windows=TRAIL_WINDOWS, k=TRAIL_K) -> dict:
    """Smoothed delay/cancel rates over the W days ending `gap` days before departure: days in (D-gap-W, D-gap].
    gap=1 is a live feed (yesterday is the newest day used); gap=90 mimics lagged BTS. Never touches a day > D-gap."""
    day = np.asarray(A["day"])[rows].astype("float64")
    end = day - gap
    ids = {n: np.asarray(A[f"{n}_id"])[rows] for n in TRAIL_GROUPS + ("g",)}
    out = {}
    for w in windows:
        raw = {n: _window(C[f"td_{n}"], ids[n], end - w, end) for n in ids}
        with np.errstate(invalid="ignore", divide="ignore"):
            for kind, num, den in (("delay", "d15", "flown"), ("cancel", "canc", "cnt")):
                rate = {"g": raw["g"][num] / raw["g"][den]}
                rate["od"] = np.nan  # set below once o and d exist
                for n in ("o", "d", "al", "nw", "rt", "oh", "fn"):
                    parent = rate["g"] if TRAIL_PARENT[n] == "g" else (
                        (rate["o"] + rate["d"]) / 2 if TRAIL_PARENT[n] == "od" else rate[TRAIL_PARENT[n]])
                    r = (raw[n][num] + k * parent) / (raw[n][den] + k)
                    rate[n] = np.where(ids[n] < 0, np.nan, r)
                    out[f"t{w}_{n}_{kind}"] = rate[n]
    return out


def level_shift_features(trail: dict, longrun: dict) -> dict:
    """90-day trailing rate minus the 12-month (booking-table) rate, for airline and origin."""
    out = {}
    for n, (dcol, ccol) in LONGRUN.items():
        out[f"shift_{n}_delay"] = np.asarray(trail[f"t90_{n}_delay"]) - np.asarray(longrun[dcol])
        out[f"shift_{n}_cancel"] = np.asarray(trail[f"t90_{n}_cancel"]) - np.asarray(longrun[ccol])
    return out


def blank_flight_number(X, mask):
    """Blank (NaN) the flight-number features on the rows in `mask`: live requests may lack a flight number."""
    cols = [c for c in X.columns if "_fn_" in c]
    X.loc[mask, cols] = np.nan
    return X


def trailing_serve_row(query: dict, C: dict, gap: int) -> dict:
    """One flight. `query` holds 1-element arrays: day and o/d/al/nw/rt/oh/fn/g ids. Same code path as the vectorised call."""
    out = trailing_features(query, C, np.array([0]), gap)
    return {k: float(v[0]) for k, v in out.items()}
