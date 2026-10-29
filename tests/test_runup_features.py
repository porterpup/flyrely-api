"""Leakage tests for the run-up backtest features: nothing after the cutoff may leak in."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backtest"))
import runup  # noqa: E402

NAN = np.nan


def flights(**cols):
    """Minimal arrays in the shape runup.load() produces."""
    A = {k: np.asarray(v) for k, v in cols.items()}
    n = len(A["s_dep"])
    A.setdefault("cancelled", np.zeros(n, dtype=bool))
    A["a_dep"] = np.where(A["cancelled"], NAN, A["s_dep"] + np.nan_to_num(A["dep_delay"]))
    A["d15"] = (np.nan_to_num(A["dep_delay"]) >= 15) & ~A["cancelled"]
    return A


def test_counter_counts_only_events_at_or_before_cutoff():
    c = runup.Counter(np.array([0, 0, 0, 1]), np.array([10.0, 20.0, 30.0, 15.0]), {"v": np.array([1, 2, 4, 8])})
    g = np.array([0, 0, 1, -1])
    t = np.array([20.0, 19.0, 100.0, 50.0])
    assert list(c.upto(g, t)[:3]) == [2, 1, 1]
    assert list(c.upto(g, t, "v")[:3]) == [3, 1, 8]
    assert np.isnan(c.upto(g, t)[3])


def test_rotation_uses_actual_delay_only_once_the_inbound_has_left():
    # One aircraft: leg 0 sched 1000 (left 50 min late), leg 1 sched 1300
    A = flights(tail_id=[7, 7], s_dep=[1000.0, 1300.0], s_arr=[1120.0, 1400.0],
                dep_delay=[50.0, 0.0], elapsed=[120.0, 100.0])
    R = runup.Rotation(A)
    rows = np.array([1])
    after = runup.rotation_features(A, R, rows, np.array([1100.0]))   # inbound left at 1050
    assert after["rot_known_delay"][0] == 50
    assert after["rot_leg_pending"][0] == 0
    before = runup.rotation_features(A, R, rows, np.array([1020.0]))  # inbound still at the gate
    assert before["rot_known_delay"][0] == 20   # only the lower bound is known
    assert before["rot_leg_pending"][0] == 1
    # slack = 1300 - 1120 - 35 = 145, so a 50 min inbound delay projects to 0
    assert after["rot_slack"][0] == 145
    assert after["rot_projected_delay"][0] == 0


def test_rotation_never_sees_its_own_flight_or_later_legs():
    A = flights(tail_id=[7, 7], s_dep=[1000.0, 1300.0], s_arr=[1120.0, 1400.0],
                dep_delay=[0.0, 90.0], elapsed=[120.0, 100.0])
    R = runup.Rotation(A)
    f = runup.rotation_features(A, R, np.array([0]), np.array([940.0]))
    assert f["rot_has_leg"][0] == 0


def test_network_pending_counts_late_flights_not_yet_gone():
    # Origin 0: flight sched 100 leaves at 160; flight sched 110 cancelled; query at 130
    A = flights(o_id=[0, 0, 0], al_id=[0, 0, 0], d_id=[1, 1, 1], s_dep=[100.0, 110.0, 400.0],
                dep_delay=[60.0, NAN, 0.0], cancelled=np.array([False, True, False]))
    C = {}
    late = np.maximum(np.nan_to_num(A["dep_delay"]), 0)
    done_t = np.where(A["cancelled"], NAN, np.fmax(A["s_dep"], A["a_dep"]))
    for name in ("o", "al"):
        gid = A[f"{name}_id"]
        C[f"dep_{name}"] = runup.Counter(gid, A["a_dep"], {"d15": A["d15"], "late": late})
        C[f"sched_{name}"] = runup.Counter(gid, A["s_dep"], {"canc": A["cancelled"], "s": A["s_dep"],
                                                             "cs": A["s_dep"] * A["cancelled"]})
        C[f"done_{name}"] = runup.Counter(gid, done_t, {"s": A["s_dep"]})
    f = runup.network_features(A, C, np.array([2]), np.array([130.0]))
    assert f["o_pending"][0] == 1
    assert f["o_pending_mean_wait"][0] == 30
    assert f["o_480m_cancelled"][0] == 1
    assert f["o_120m_departed"][0] == 0      # the late flight hasn't left yet at 130
    later = runup.network_features(A, C, np.array([2]), np.array([170.0]))
    assert later["o_pending"][0] == 0
    assert later["o_120m_delayed_share"][0] == 1
