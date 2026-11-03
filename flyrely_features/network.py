"""Same-day network state at the cutoff: origin, destination and whole-airline departures, delays, cancellations, backlog."""

import numpy as np


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


def serve_row(query: dict, C: dict, c: float) -> dict:
    """One flight at cutoff `c` (Eastern-reference minutes). `query` holds 1-element o_id/d_id/al_id arrays."""
    out = network_features(query, C, np.array([0]), np.array([float(c)]))
    return {k: float(v[0]) for k, v in out.items()}
