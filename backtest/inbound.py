"""Inbound-aircraft state at a cutoff, and rotation pairings (A7 / L0, backtest/PROTOCOL_A7_L0.md).

`true_inbound` uses the tail BTS recorded (known only after the fact: an upper bound on what linkage can deliver).
`prior_inbound` predicts the inbound from earlier days' rotations only, with no tail. `inbound_features` computes the
state of a given inbound leg using only times at or before the cutoff.
"""

import numpy as np

from flyrely_features._common import MIN_TURN
from flyrely_features.rotation import Rotation

MAX_GAP = 18 * 60          # an inbound must have been scheduled to leave within 18 h before this flight
DAY_STRIDE = 5000          # larger than the number of days in the data
IN_COLS = ["in_departed", "in_cancelled_known", "in_dep_delay", "in_proj_arr", "in_slack", "in_proj_late",
           "in_eta_passed", "in_through"]


def true_inbound(A, R: Rotation, rows=None) -> np.ndarray:
    """Row index of the immediate predecessor leg of the same tail that ends at this flight's origin (-1 if none)."""
    rows = np.arange(len(A["s_dep"])) if rows is None else np.asarray(rows)
    pos = R.pos[rows]
    has = pos > 0
    prev = R.rows[np.clip(pos - 1, 0, len(R.rows) - 1)]
    ok = (has & (A["tail_id"][prev] == A["tail_id"][rows]) & (A["d_id"][prev] == A["o_id"][rows])
          & (A["s_dep"][prev] > A["s_dep"][rows] - MAX_GAP) & (A["s_dep"][prev] < A["s_dep"][rows]))
    return np.where(ok, prev, -1)


def inbound_features(A, rows, prev, c) -> dict:
    """State of inbound leg `prev` (row indices, -1 = unknown) as known at cutoff c. NaN where there is no inbound."""
    rows, c = np.asarray(rows), np.asarray(c, dtype="float64")
    valid = prev >= 0
    p = np.where(valid, prev, 0)
    s_p, a_p, el = A["s_dep"][p], A["a_dep"][p], A["elapsed"][p]
    canc = A["cancelled"][p]
    departed = ~canc & (a_p <= c)                       # actual departure at or before the cutoff
    canc_known = canc & (s_p <= c)                      # a cancellation is visible once its departure time has passed
    dep_delay = np.where(departed, np.nan_to_num(A["dep_delay"][p]), np.maximum(c - s_p, 0))
    proj_dep = np.where(departed, a_p, np.maximum(c, s_p))
    proj_arr = proj_dep + el                            # no BTS ArrDelay: ETA = departure + scheduled block time
    slack = A["s_dep"][rows] - proj_arr - MIN_TURN
    nan = lambda x: np.where(valid & ~canc_known, x, np.nan)
    return {
        "in_departed": nan(departed.astype(float)),
        "in_cancelled_known": np.where(valid, canc_known.astype(float), np.nan),
        "in_dep_delay": nan(dep_delay),
        "in_proj_arr": nan(proj_arr - A["s_dep"][rows]),
        "in_slack": nan(slack),
        "in_proj_late": nan(np.maximum(-slack, 0)),
        "in_eta_passed": nan((proj_arr <= c).astype(float)),
        "in_through": np.where(valid, (A["fn_id"][p] == A["fn_id"][rows]).astype(float), np.nan),
    }


# --------------------------- pairings (L0 and the prior linkage) ---------------------------

def _lookup(sorted_keys: np.ndarray, vals: np.ndarray, q: np.ndarray) -> np.ndarray:
    """vals at q (exact match in sorted_keys), -1 where absent."""
    if len(sorted_keys) == 0:
        return np.full(len(q), -1, dtype=np.int64)
    i = np.clip(np.searchsorted(sorted_keys, q), 0, len(sorted_keys) - 1)
    return np.where(sorted_keys[i] == q, vals[i], -1)


class Pairings:
    """Pairing table: (flight number B, origin, day) -> (inbound flight number A, day offset), from true tails."""

    def __init__(self, A, R: Rotation):
        self.A = A
        n_ap = int(max(A["o_id"].max(), A["d_id"].max())) + 1
        self.n_ap = n_ap
        self.prev = true_inbound(A, R)
        b = np.flatnonzero(self.prev >= 0)
        a = self.prev[b]
        self.b_key = (A["fn_id"][b].astype(np.int64) * n_ap + A["o_id"][b]) * DAY_STRIDE + A["day"][b]
        order = np.argsort(self.b_key, kind="stable")
        b, a = b[order], a[order]
        keys = self.b_key[order]
        first = np.concatenate([[True], keys[1:] != keys[:-1]]) if len(keys) else np.array([], bool)
        self.keys = keys[first]
        self.fn_a = A["fn_id"][a][first]
        self.off = (A["day"][b] - A["day"][a])[first]
        # every leg, keyed (flight number, destination, day) -> row, for finding the predicted inbound on a given day
        legs = np.arange(len(A["day"]))
        lk = (A["fn_id"].astype(np.int64) * n_ap + A["d_id"]) * DAY_STRIDE + A["day"]
        o = np.argsort(lk, kind="stable")
        lk, legs = lk[o], legs[o]
        f = np.concatenate([[True], lk[1:] != lk[:-1]])
        self.leg_keys, self.leg_rows = lk[f], legs[f]

    def fn_a_on(self, rows, day):
        """Inbound flight number (-1 if none) and day offset for flights `rows` as they flew on `day` (array)."""
        A = self.A
        q = (A["fn_id"][rows].astype(np.int64) * self.n_ap + A["o_id"][rows]) * DAY_STRIDE + day
        i = np.clip(np.searchsorted(self.keys, q), 0, max(len(self.keys) - 1, 0))
        hit = self.keys[i] == q
        return np.where(hit, self.fn_a[i], -1), np.where(hit, self.off[i], 0)

    def leg_row(self, fn_a, dest, day):
        q = (np.where(fn_a >= 0, fn_a, 0).astype(np.int64) * self.n_ap + dest) * DAY_STRIDE + day
        r = _lookup(self.leg_keys, self.leg_rows, q)
        return np.where(fn_a >= 0, r, -1)

    def predicted(self, rows, last_day, lags=(7, 1)) -> tuple[np.ndarray, np.ndarray]:
        """Predicted inbound flight number per flight from the earliest-listed lag that exists, only from days
        <= last_day (array, the day before the cutoff day). Returns (fn_a, day offset)."""
        A = self.A
        fn = np.full(len(rows), -1, dtype=np.int64)
        off = np.zeros(len(rows), dtype=np.int64)
        for lag in lags:
            d = A["day"][rows] - lag
            f, o = self.fn_a_on(rows, d)
            use = (fn < 0) & (f >= 0) & (d <= last_day)
            fn, off = np.where(use, f, fn), np.where(use, o, off)
        return fn, off

    def prior_inbound(self, rows, last_day, lags=(7, 1)) -> np.ndarray:
        """Row of the predicted inbound leg for each flight (-1 if none): the predicted flight number as it flies today."""
        A = self.A
        fn, off = self.predicted(rows, last_day, lags)
        r = self.leg_row(fn, A["o_id"][rows], A["day"][rows] - off)
        ok = (r >= 0)
        rr = np.where(ok, r, 0)
        ok &= (A["s_dep"][rr] > A["s_dep"][rows] - MAX_GAP) & (A["s_dep"][rr] < A["s_dep"][rows])
        return np.where(ok, r, -1)


def plurality_prev_weeks(P: Pairings, rows, weeks=4) -> np.ndarray:
    """Most common inbound flight number over the same flight on the last `weeks` same weekdays (ties: most recent)."""
    cands = np.stack([P.fn_a_on(rows, P.A["day"][rows] - 7 * k)[0] for k in range(1, weeks + 1)], axis=1)
    best = np.full(len(rows), -1, dtype=np.int64)
    best_n = np.zeros(len(rows), dtype=np.int64)
    for k in range(weeks):                    # recent first; strict > keeps the most recent on ties
        v = cands[:, k]
        n = ((cands == v[:, None]) & (v[:, None] >= 0)).sum(1)
        upd = (v >= 0) & (n > best_n)
        best, best_n = np.where(upd, v, best), np.where(upd, n, best_n)
    return best
