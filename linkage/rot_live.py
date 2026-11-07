"""The nine rot_* features computed from an explicit chain of legs (L1 plan, sections 2.3 and 2.5).

`flyrely_features.rotation.rotation_features` (frozen) finds the chain itself from BTS tail numbers. Live, the chain comes
from linkage rules (M1/M2/M3), so this module takes the chain as an input and applies the same arithmetic.

- `strict_asof=False` and the BTS true chain (`true_chain`): equals `rotation_features` exactly.
- `strict_asof=True`: a cancellation counts only if it was received by the as-of time `c` (`cancel_recv`). This is the
  version used for L1-sim training and for serving. Only features of a chain's latest leg that is cancelled can differ.

`A` is the schedule-array dict of `backtest/runup.load` (s_dep, s_arr, a_dep, dep_delay, elapsed, cancelled, tail_id), all
in minutes on one clock; `rows` are the departing legs B; `c` is the as-of time (same clock) per row.
"""

import numpy as np

from flyrely_features._common import MIN_TURN
from flyrely_features.rotation import Rotation

MAX_GAP = 18 * 60
ROT_COLS = ["rot_has_leg", "rot_known_delay", "rot_leg_pending", "rot_leg_cancelled", "rot_legs_between",
            "rot_slack", "rot_projected_delay", "rot_minutes_since_leg_sched", "rot_sched_turn"]


def true_chain(A, R: Rotation, rows, c=None) -> np.ndarray:
    """BTS true chain of each row: the same tail's earlier legs scheduled after `min(B's sched OUT, c) - 18 h`, as an
    (n, K) int array of A row indices, oldest first, the immediate predecessor last, -1 padded on the left. K is the
    longest chain present. `c` (as-of times) widens the window to what rot_* can see (a leg within 18 h of c)."""
    rows = np.asarray(rows)
    pos = R.pos[rows]
    s_b, tail = A["s_dep"][rows], A["tail_id"][rows]
    lo = (s_b if c is None else np.minimum(s_b, np.asarray(c, dtype="float64"))) - MAX_GAP
    alive = pos >= 0
    cols, k = [], 1
    while alive.any():
        p = pos - k
        leg = R.rows[np.clip(p, 0, len(R.rows) - 1)]
        alive = alive & (p >= 0) & (A["tail_id"][leg] == tail) & (A["s_dep"][leg] > lo)
        cols.append(np.where(alive, leg, -1))
        k += 1
    while cols and not (cols[-1] >= 0).any():
        cols.pop()
    if not cols:
        return np.full((len(rows), 1), -1, dtype=np.int64)
    return np.stack(cols[::-1], axis=1).astype(np.int64)


def rot_live(A, rows, c, chain, strict_asof: bool = False, cancel_recv=None) -> dict:
    """The nine rot_* features for departing legs `rows` at as-of times `c`, from `chain` ((n, K) A-row indices, oldest
    first, immediate predecessor last, -1 padding on the left).

    `cancel_recv` (per A row, same clock as `c`) is when a leg's cancellation became known; used only with
    `strict_asof=True`. Default is the leg's scheduled departure (as in backtest/inbound.py: a cancellation is visible
    once the departure time has passed)."""
    rows = np.asarray(rows)
    c = np.asarray(c, dtype="float64")
    chain = np.asarray(chain, dtype=np.int64)
    if chain.ndim == 1:
        chain = chain[:, None]
    n, K = chain.shape
    present = chain >= 0
    ch = np.where(present, chain, 0)

    s_b = A["s_dep"][rows]
    fc = np.floor(np.nan_to_num(c))
    # slots at or before the as-of time (the frozen code compares whole-minute keys); older slots come first
    at_c = present & (np.floor(np.nan_to_num(A["s_dep"][ch])) <= fc[:, None])
    anyc = at_c.any(1)
    jpos = K - 1 - np.argmax(at_c[:, ::-1], axis=1)
    leg = ch[np.arange(n), jpos]
    leg_present = present[np.arange(n), jpos] & anyc
    # a chain that skips a slot between the leg and B (a -1 hole) is not a chain
    after = np.arange(K)[None, :] > jpos[:, None]
    contiguous = ~((~present) & after).any(1)

    valid = (leg_present & contiguous & ~np.isnan(s_b) & (np.floor(s_b) > fc)
             & (A["s_dep"][leg] > c - MAX_GAP))

    s_leg, a_leg = A["s_dep"][leg], A["a_dep"][leg]
    departed = a_leg <= c
    known = np.where(departed, np.nan_to_num(A["dep_delay"][leg]), np.maximum(c - s_leg, 0))
    leg_cancelled = np.asarray(A["cancelled"])[leg]
    if strict_asof:
        recv = A["s_dep"] if cancel_recv is None else np.asarray(cancel_recv)
        leg_cancelled = leg_cancelled & (recv[leg] <= c)
    known = np.where(leg_cancelled, np.nan, known)

    between = np.where(valid, K - 1 - jpos, 0)
    block = np.where(after & present, A["elapsed"][ch], 0.0).sum(1)
    block = np.where(between > 0, block, 0.0)
    slack = s_b - A["s_arr"][leg] - block - (between + 1) * MIN_TURN

    prev = chain[:, -1]
    pv = np.where(prev >= 0, prev, 0)
    prev_ok = (prev >= 0) & (A["s_dep"][pv] > s_b - MAX_GAP)

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
        "rot_sched_turn": np.where(prev_ok, s_b - A["s_arr"][pv], np.nan),
    }
