"""Aircraft rotation: where this flight's aircraft is at the cutoff and whether it can get here on time."""

import numpy as np

from ._common import MIN_TURN, T


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


def serve_row(A, R: Rotation, row: int, c: float) -> dict:
    """One flight, identified by its index in the schedule arrays `A` the Rotation was built from."""
    out = rotation_features(A, R, np.array([row]), np.array([float(c)]))
    return {k: float(v[0]) for k, v in out.items()}
