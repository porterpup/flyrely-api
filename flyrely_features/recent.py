"""Trailing 7-day recent performance (delay/cancel rate and volume) for origin, destination, airline, route, flight number."""

import numpy as np

from ._common import cutoff_day


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
