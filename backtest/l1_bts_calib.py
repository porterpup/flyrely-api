"""L1 / T4: offline calibration of the linkage rules on BTS (plan: backtest/PLAN_L1_LIVE_TAIL_LINKAGE.md, section 4, T4).

    python backtest/l1_bts_calib.py     # writes backtest/results/l1_bts_calib.json

Validation months 2026-01..08 only (evaluation rows). Rows from 2025-12 are loaded as history for the d-7 / d-1 pairing
lookups. Nothing from 2026-09 on is loaded and no holdout unlock is used. Offline, $0.

Measures, with the BTS true tail as ground truth:
  * M2 (through-flight: the same flight number arrives at B's origin within 18 h before B): precision by carrier.
  * Chain length (links from B back to the aircraft's latest leg already scheduled to have left) at T-3h and T-1h,
    using the true chain through `linkage.rot_live` (the upper bound live linkage can reach).
  * M3 (weekday prior: d-7, else d-1) precision, with the exclusion rule, which is simulated: a share p of departures
    is "M1-known" (tail known from its own flight plan); a candidate inbound whose next departure is M1-known and is
    not B is removed. p = 0 (no exclusion), 0.10 (about T-3h fill) and 0.65 (about T-1h fill).

BTS has no aircraft-type column, so the "same type family" filter in rules M2 and M3 cannot be measured here; every
precision below is WITHOUT the type filter (a lower bound on what the filter delivers). It is measured on the live archive.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import backtest as bt  # noqa: E402
import runup  # noqa: E402
from flyrely_features._common import T  # noqa: E402
from flyrely_features.rotation import Rotation  # noqa: E402
from inbound import MAX_GAP, Pairings, true_inbound  # noqa: E402
from linkage.rot_live import rot_live, true_chain  # noqa: E402

RESULTS = HERE / "results"
HISTORY_FROM = "2025-12"
EVAL = ("2026-01", "2026-08")
FORBIDDEN_FROM = pd.Timestamp("2026-09-01")
HORIZONS = {"3h": 180, "1h": 60}
M1_SHARE = {"0": 0.0, "0.10": 0.10, "0.65": 0.65}      # M1-known share of departures (plan section 1: ~T-3h, ~T-1h)
SEED = 42
MIN_CARRIER_FLIGHTS = 20000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_window(start: str = HISTORY_FROM, end: str = EVAL[1]):
    """runup.load for months start..end (runup.load itself always starts at 2024-01); `end` must be a validation month."""
    if pd.Period(end, "M") >= pd.Period("2026-09", "M"):
        raise RuntimeError("protected holdout month requested")
    orig = bt.load_months
    bt.load_months = lambda s, e, holdout_dir=None: orig(start, e, holdout_dir)
    try:
        df, A = runup.load(end=end)
    finally:
        bt.load_months = orig
    if df["FlightDate"].max() >= FORBIDDEN_FROM:
        raise RuntimeError("protected holdout data (>= 2026-09) present")
    return df, A


def ratio(a, b, nd=4):
    return round(float(a) / float(b), nd) if b else None


def m2_candidates(A, rows) -> np.ndarray:
    """Per row B: the latest earlier leg of the same flight number arriving at B's origin, scheduled to leave within
    18 h before B and to arrive before B leaves (-1 if none)."""
    n_ap = len(A["airports"])
    leg_key = (A["fn_id"].astype(np.int64) * n_ap + A["d_id"]) * T + np.floor(A["s_dep"]).astype(np.int64)
    order = np.argsort(leg_key, kind="stable")
    sk = leg_key[order]
    grp = (A["fn_id"][rows].astype(np.int64) * n_ap + A["o_id"][rows])
    q = grp * T + np.floor(A["s_dep"][rows]).astype(np.int64)
    i = np.searchsorted(sk, q, side="left") - 1          # strictly earlier whole-minute departure
    ii = np.clip(i, 0, len(sk) - 1)
    cand = order[ii]
    ok = ((i >= 0) & (sk[ii] // T == grp) & (A["s_dep"][cand] > A["s_dep"][rows] - MAX_GAP)
          & (A["s_arr"][cand] <= A["s_dep"][rows]) & (cand != rows))
    return np.where(ok, cand, -1)


def successor_table(prev_of: np.ndarray, n: int) -> np.ndarray:
    """succ[a] = the leg whose true inbound is a (the next departure of a's aircraft), -1 if none."""
    succ = np.full(n, -1, dtype=np.int64)
    b = np.flatnonzero(prev_of >= 0)
    succ[prev_of[b]] = b
    return succ


def excluded(cand, rows, succ, m1_known) -> np.ndarray:
    """Exclusion rule: candidate inbound whose next departure is M1-known (tail filed) and is not this flight."""
    s = succ[np.where(cand >= 0, cand, 0)]
    return (cand >= 0) & (s >= 0) & (s != rows) & m1_known[np.where(s >= 0, s, 0)]


def score_candidates(cand, rows, true_in, tail, have_true) -> dict:
    has = cand >= 0
    right = has & (cand == true_in[rows])
    same_tail = has & (tail[np.where(has, cand, 0)] == tail[rows]) & (tail[rows] >= 0)
    g = has & have_true
    return {"n": int(len(rows)), "coverage": ratio(has.sum(), len(rows)),
            "precision": ratio(right.sum(), has.sum()),
            "precision_same_tail": ratio(same_tail.sum(), has.sum()),
            "precision_given_pairing": ratio((right & have_true).sum(), g.sum()),
            "n_candidates": int(has.sum()),
            "hit_over_all": ratio(right.sum(), len(rows))}


def chain_length_dist(A, R, df, ev) -> dict:
    out = {}
    regional = (df["Reporting_Airline"].to_numpy() != df["Marketing_Airline_Network"].to_numpy())
    labels = ["no_tail", "no_leg", "1", "2", "3", "4", "5+"]
    for h, mins in HORIZONS.items():
        counts = {seg: np.zeros(len(labels), dtype=np.int64) for seg in ("all", "mainline", "regional")}
        for lo in range(0, len(ev), 400_000):
            rows = ev[lo:lo + 400_000]
            c = A["s_dep"][rows] - mins
            f = rot_live(A, rows, c, true_chain(A, R, rows, c))
            has = f["rot_has_leg"] == 1
            links = np.where(has, f["rot_legs_between"] + 1, 0).astype(int)
            cls = np.where(A["tail_id"][rows] < 0, 0, np.where(~has, 1, 1 + np.minimum(links, 5)))
            reg = regional[rows]
            for seg, m in (("all", np.ones(len(rows), bool)), ("mainline", ~reg), ("regional", reg)):
                counts[seg] += np.bincount(cls[m], minlength=len(labels))[:len(labels)]
        out[h] = {}
        for seg, cnt in counts.items():
            tot = int(cnt.sum())
            out[h][seg] = {"n": tot, "share": {l: ratio(v, tot) for l, v in zip(labels, cnt)},
                           "mean_links_when_linked": ratio(sum(int(l) * v for l, v in zip(labels[2:6], cnt[2:6]))
                                                           + 5 * int(cnt[6]), int(cnt[2:].sum()), 3)}
        log("chain length", h, out[h]["all"]["share"])
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    df, A = load_window()
    n = len(A["s_dep"])
    ev = np.flatnonzero(((df["ym"] >= EVAL[0]) & (df["ym"] <= EVAL[1])).to_numpy())
    log(f"loaded {n:,} rows, evaluating {len(ev):,} (months {EVAL[0]}..{EVAL[1]})")
    R = Rotation(A)
    prev_of = true_inbound(A, R)
    have_true = prev_of[ev] >= 0
    tail = A["tail_id"]
    carrier = df["Reporting_Airline"].to_numpy()[ev]
    regional = (df["Reporting_Airline"].to_numpy() != df["Marketing_Airline_Network"].to_numpy())[ev]
    succ = successor_table(prev_of, n)
    u = np.random.default_rng(SEED).random(n)
    res = {"plan": "backtest/PLAN_L1_LIVE_TAIL_LINKAGE.md T4", "eval_months": list(EVAL), "history_from": HISTORY_FROM,
           "n_eval_flights": int(len(ev)), "n_eval_with_true_inbound": int(have_true.sum()),
           "type_filter": "not measurable on BTS (no aircraft type column); all precisions are without it",
           "m1_share_scenarios": M1_SHARE, "seed": SEED}

    # ---- chain length -------------------------------------------------------------
    res["chain_length"] = chain_length_dist(A, R, df, ev)

    # ---- M2 -----------------------------------------------------------------------
    cand2 = m2_candidates(A, ev)
    m2 = {"definition": "same reporting carrier + flight number arrives at B's origin, sched OUT within 18 h before B, "
                        "sched IN <= B's sched OUT, latest such leg; precision = candidate is B's true immediate inbound "
                        "(precision_same_tail: candidate has B's tail)"}
    base = score_candidates(cand2, ev, prev_of, tail, have_true)
    # the true through-flight share (the flight number really continues) for context
    base["through_share_of_true_pairs"] = ratio((A["fn_id"][np.where(have_true, prev_of[ev], 0)] == A["fn_id"][ev])[have_true].sum(),
                                                have_true.sum())
    m2["overall"] = base
    m2["mainline"] = score_candidates(cand2[~regional], ev[~regional], prev_of, tail, have_true[~regional])
    m2["regional"] = score_candidates(cand2[regional], ev[regional], prev_of, tail, have_true[regional])
    counts = pd.Series(carrier).value_counts()
    m2["by_carrier"] = {}
    for cr in counts[counts >= MIN_CARRIER_FLIGHTS].index:
        m = carrier == cr
        m2["by_carrier"][cr] = score_candidates(cand2[m], ev[m], prev_of, tail, have_true[m])
    m2["with_exclusion"] = {}
    for name, p in M1_SHARE.items():
        m1 = u < p
        keep = ~m1[ev]                                             # M1-known flights do not use M2
        c2 = np.where(excluded(cand2, ev, succ, m1), -1, cand2)
        m2["with_exclusion"][name] = score_candidates(c2[keep], ev[keep], prev_of, tail, have_true[keep])
    res["m2"] = m2
    log("M2 overall", base)

    # ---- M3 -----------------------------------------------------------------------
    P = Pairings(A, R)
    last = A["day"][ev] - 1
    c7 = P.prior_inbound(ev, last, lags=(7,))
    c1 = P.prior_inbound(ev, last, lags=(1,))
    # reproduction of the L0 number (flight-number match given a pairing existed on d-7)
    pred7, _ = P.fn_a_on(ev, A["day"][ev] - 7)
    today_fn = np.where(have_true, A["fn_id"][np.where(have_true, prev_of[ev], 0)], -1)
    ex = have_true & (pred7 >= 0)
    m3 = {"l0_baseline_reported": 0.574,
          "l0_reproduced_fn_match_given_exists": ratio((ex & (pred7 == today_fn)).sum(), ex.sum()),
          "definition": "candidate = the leg flying today's d-7 inbound flight number into B's origin (else d-1), "
                        "scheduled within 18 h before B; precision = candidate is B's true immediate inbound "
                        "(row identity, stricter than the L0 flight-number match)"}
    m3["scenarios"] = {}
    for name, p in M1_SHARE.items():
        m1 = u < p
        keep = ~m1[ev]
        e7, e1 = excluded(c7, ev, succ, m1), excluded(c1, ev, succ, m1)
        a7 = np.where(e7, -1, c7)
        a1 = np.where(e1, -1, c1)
        both = np.where(a7 >= 0, a7, a1)
        sc = {}
        for label, cand in (("d-7", a7), ("d-7_else_d-1", both)):
            sc[label] = {"overall": score_candidates(cand[keep], ev[keep], prev_of, tail, have_true[keep]),
                         "mainline": score_candidates(cand[keep & ~regional], ev[keep & ~regional], prev_of, tail,
                                                      have_true[keep & ~regional]),
                         "regional": score_candidates(cand[keep & regional], ev[keep & regional], prev_of, tail,
                                                      have_true[keep & regional])}
        sc["share_of_d7_candidates_removed"] = ratio((e7 & keep).sum(), ((c7 >= 0) & keep).sum())
        m3["scenarios"][name] = sc
        log("M3 p_m1", name, sc["d-7_else_d-1"]["overall"])
    res["m3"] = m3

    (RESULTS / "l1_bts_calib.json").write_text(json.dumps(res, indent=2))
    log("wrote", RESULTS / "l1_bts_calib.json")


if __name__ == "__main__":
    main()
