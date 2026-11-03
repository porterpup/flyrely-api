"""Constants and the time-cutoff counter shared by the run-up feature groups."""

import numpy as np
import pandas as pd

HORIZONS = {"7d": 7 * 1440, "1d": 1440, "3h": 180, "1h": 60}  # minutes before departure
SAME_DAY = {"3h", "1h"}
MIN_TURN = 35            # minutes an aircraft needs on the ground between legs
SHIFT = 2000             # keeps every clock time positive (Guam is UTC+10)
T = 2_000_000            # key stride per group; larger than any time in the data
DAY0 = pd.Timestamp("2024-01-01")
WEATHER_VARS = ["precipitation", "snowfall", "wind_gusts_10m", "cape", "weather_code"]


def cutoff_day(A, rows, h):
    return A["day"][rows] + np.floor((A["dep_local"][rows] - HORIZONS[h]) / 1440)


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
