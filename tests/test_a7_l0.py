"""Offline tests for the A7/L0 inbound-aircraft features and pairing tables (no data files needed)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backtest"))
import a7_l0  # noqa: E402
from inbound import Pairings, inbound_features, plurality_prev_weeks, true_inbound  # noqa: E402
from flyrely_features.rotation import Rotation  # noqa: E402

NAN = np.nan


def sched(**cols):
    A = {k: np.asarray(v) for k, v in cols.items()}
    n = len(A["s_dep"])
    A.setdefault("cancelled", np.zeros(n, dtype=bool))
    A.setdefault("day", np.zeros(n, dtype=np.int64))
    A["a_dep"] = np.where(A["cancelled"], NAN, A["s_dep"] + np.nan_to_num(A["dep_delay"]))
    A["airports"] = np.array(["AAA", "BBB", "CCC"])
    return A


def two_legs(dep_delay0=50.0):
    # tail 7: leg0 AAA->BBB sched 1000 (block 120), leg1 BBB->CCC sched 1300
    return sched(tail_id=[7, 7], s_dep=[1000.0, 1300.0], s_arr=[1120.0, 1400.0], dep_delay=[dep_delay0, 0.0],
                 elapsed=[120.0, 100.0], o_id=[0, 1], d_id=[1, 2], fn_id=[10, 11])


def test_true_inbound_is_the_previous_leg_ending_at_the_origin():
    A = two_legs()
    R = Rotation(A)
    assert list(true_inbound(A, R)) == [-1, 0]
    A["d_id"] = np.array([2, 2])                      # previous leg ends elsewhere: no inbound
    assert true_inbound(A, Rotation(A))[1] == -1


def test_inbound_state_uses_only_what_is_known_at_the_cutoff():
    A = two_legs(dep_delay0=50.0)                     # inbound leaves at 1050
    rows, prev = np.array([1]), np.array([0])
    before = inbound_features(A, rows, prev, np.array([1020.0]))   # not yet left: only a lower bound, schedule block
    assert before["in_departed"][0] == 0
    assert before["in_dep_delay"][0] == 20             # c - sched; the real 50 is not visible
    assert before["in_proj_arr"][0] == 1020 + 120 - 1300
    after = inbound_features(A, rows, prev, np.array([1100.0]))    # left at 1050
    assert after["in_departed"][0] == 1 and after["in_dep_delay"][0] == 50
    assert after["in_proj_arr"][0] == 1050 + 120 - 1300
    assert after["in_slack"][0] == 1300 - 1170 - 35


def test_features_before_cutoff_do_not_change_when_the_future_changes():
    c = np.array([1020.0])
    f1 = inbound_features(two_legs(30.0), np.array([1]), np.array([0]), c)
    f2 = inbound_features(two_legs(200.0), np.array([1]), np.array([0]), c)   # inbound's later delay differs
    for k in f1:
        assert np.array_equal(f1[k], f2[k], equal_nan=True), k


def test_cancellation_is_visible_only_after_the_scheduled_departure():
    A = two_legs()
    A["cancelled"] = np.array([True, False])
    A["a_dep"][0] = NAN
    early = inbound_features(A, np.array([1]), np.array([0]), np.array([900.0]))
    late = inbound_features(A, np.array([1]), np.array([0]), np.array([1100.0]))
    assert early["in_cancelled_known"][0] == 0 and not np.isnan(early["in_slack"][0])
    assert late["in_cancelled_known"][0] == 1 and np.isnan(late["in_slack"][0])


def test_no_inbound_gives_nan():
    A = two_legs()
    f = inbound_features(A, np.array([0]), np.array([-1]), np.array([900.0]))
    assert all(np.isnan(v[0]) for v in f.values())


def weekly_rotation(n_weeks=3):
    """Flight 10 (AAA->BBB) then flight 11 (BBB->CCC) every day at the same times; one tail per day."""
    rows = dict(tail_id=[], s_dep=[], s_arr=[], dep_delay=[], elapsed=[], o_id=[], d_id=[], fn_id=[], day=[])
    for d in range(7 * n_weeks):
        for fn, o, dd, s in ((10, 0, 1, 600.0), (11, 1, 2, 900.0)):
            rows["tail_id"].append(d); rows["s_dep"].append(d * 1440 + s); rows["s_arr"].append(d * 1440 + s + 120)
            rows["dep_delay"].append(0.0); rows["elapsed"].append(120.0); rows["o_id"].append(o); rows["d_id"].append(dd)
            rows["fn_id"].append(fn); rows["day"].append(d)
    return sched(**rows)


def test_pairing_table_and_prior_use_only_earlier_days():
    A = weekly_rotation()
    R = Rotation(A)
    P = Pairings(A, R)
    outbound = np.flatnonzero(A["fn_id"] == 11)
    day14 = outbound[A["day"][outbound] == 14]
    # with data only up to day 13 the d-7 pairing (flight 10) predicts the inbound for day 14
    r = P.prior_inbound(day14, last_day=np.array([13]))
    assert A["fn_id"][r[0]] == 10 and A["day"][r[0]] == 14
    # if the data allowed ends before day 7 (the d-7 day for day 14), there is nothing to predict from
    none = P.prior_inbound(day14, last_day=np.array([5]), lags=(7,))
    assert none[0] == -1


def test_plurality_and_stability_rate():
    A = weekly_rotation(n_weeks=5)
    P = Pairings(A, Rotation(A))
    outbound = np.flatnonzero(A["fn_id"] == 11)
    late = outbound[A["day"][outbound] >= 28]
    assert (plurality_prev_weeks(P, late, 4) == 10).all()
    have = P.prev[late] >= 0
    today = A["fn_id"][P.prev[late]]
    f, _ = P.fn_a_on(late, A["day"][late] - 7)
    out = a7_l0.rate(f == today, f >= 0, have)
    assert out["p_match_given_exists"] == 1.0 and out["coverage"] == 1.0


def test_link_mask_is_nested_and_regional_never_linked():
    u = np.random.default_rng(0).random(10_000)
    a, b = a7_l0.link_mask(u, 0.29), a7_l0.link_mask(u, 0.5)
    assert (a <= b).all() and abs(a.mean() - 0.29) < 0.02
    reg = np.arange(10_000) % 2 == 0
    m = a7_l0.link_mask(u, 1.0, reg)
    assert not m[reg].any() and m[~reg].all()


def test_decision_rules():
    def mk(perfect, ci_lo, g29, g29_lo, gap, gap_lo, pr, pr_lo, p7):
        return {"3h": {"gain_vs_N": {"P+": {"diff": perfect, "ci95": [ci_lo, perfect]},
                                     "L0.29": {"diff": g29, "ci95": [g29_lo, g29]},
                                     "PR": {"diff": pr, "ci95": [pr_lo, pr]}},
                       "perfect_minus_level": {"0.5": {"gain_of_perfect": gap, "ci95": [gap_lo, gap]}}},
                "l0": {"overall": {"d-7": {"p_match_given_exists": p7}}}}
    d = a7_l0.decide(mk(0.0003, 0.0001, 0.0, -0.0001, 0.0, -0.0001, 0.0, 0.0, 0.7))
    assert d["linkage_priority"].startswith("no") and not d["paid_justified_vs_0.5"]
    d = a7_l0.decide(mk(0.01, 0.009, 0.003, 0.002, 0.004, 0.003, 0.006, 0.005, 0.7))
    assert d["linkage_priority"].startswith("yes") and d["sfdps_level_useful"] and d["paid_justified_vs_0.5"]
    assert d["prior_worth_building"] and d["prior_strong"]
    assert not a7_l0.decide(mk(0.01, 0.009, 0.003, 0.002, 0.004, 0.003, 0.006, 0.005, 0.5))["prior_worth_building"]
