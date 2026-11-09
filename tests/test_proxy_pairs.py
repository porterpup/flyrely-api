"""Proxy-pair builder on synthetic legs (no data files)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
sys.path.insert(0, str(ROOT))
import pairs  # noqa: E402

MCT = {"DEN": 40}
HUBS = ("DEN",)


def leg(origin, dest, dep, arr, net="UA", delay_dep=0.0, delay_arr=0.0, fn=1, tail="", cancelled=False, diverted=False,
        carrier=None, fn_op=None):
    dep, arr = pd.Timestamp(dep), pd.Timestamp(arr)
    return {"network": net, "carrier": carrier or net, "fn_marketing": fn, "fn_operating": fn if fn_op is None else fn_op,
            "tail": tail, "origin": origin, "dest": dest, "flight_date": dep.normalize(), "dep_wall": dep,
            "arr_wall": arr, "cancelled": cancelled, "diverted": diverted,
            "dep_delay": np.nan if cancelled else delay_dep,
            "arr_delay": np.nan if cancelled else (np.inf if diverted else delay_arr), "ym": dep.strftime("%Y-%m")}


def frame(*legs):
    return pd.DataFrame(list(legs))


def build(legs, **kw):
    return pairs.build_pairs(legs, legs, MCT, hubs=HUBS, **kw)


# SFO->DEN arrives 10:00 ; DEN->ORD leaves 11:00 (layover 60). SFO-DEN 967mi, DEN-ORD 888mi, SFO-ORD 1846: detour ~1.0
A = leg("SFO", "DEN", "2026-03-02 07:00", "2026-03-02 10:00", fn=1, delay_arr=30)
B = leg("DEN", "ORD", "2026-03-02 11:00", "2026-03-02 14:00", fn=2, delay_dep=0)


def test_basic_pair_and_miss_boundary():
    r = build(frame(A, B))
    assert len(r) == 1 and r.layover[0] == 60 and r.M[0] == 40
    assert bool(r.miss[0]) is True                       # X=30 > s=20
    # X - Y == s is not a miss: delay 20 with s = 20
    r = build(frame({**A, "arr_delay": 20.0}, B))
    assert bool(r.miss[0]) is False
    r = build(frame({**A, "arr_delay": 21.0}, B))
    assert bool(r.miss[0]) is True
    r = build(frame({**A, "arr_delay": 41.0}, {**B, "dep_delay": 21.0}))   # 41-21 = 20 == s: made it
    assert bool(r.miss[0]) is False
    r = build(frame({**A, "arr_delay": 42.0}, {**B, "dep_delay": 21.0}))
    assert bool(r.miss[0]) is True


def test_diverted_leg1_is_a_miss_and_cancellations_are_not_misses():
    r = build(frame({**A, "diverted": True, "arr_delay": np.inf}, B))
    assert bool(r.miss[0]) and bool(r.disrupted[0])
    r = build(frame({**A, "cancelled": True, "arr_delay": np.nan, "dep_delay": np.nan}, B))
    assert not bool(r.miss[0]) and bool(r.c1[0]) and bool(r.disrupted[0])
    r = build(frame(A, {**B, "cancelled": True, "dep_delay": np.nan}))
    assert not bool(r.miss[0]) and bool(r.c2[0]) and bool(r.disrupted[0])


@pytest.mark.parametrize("b_override", [
    {"network": "DL", "carrier": "DL"},                  # different network
    {"network": "NK", "carrier": "NK"},
    {"dest": "SFO"},                                     # back to where leg 1 started
    {"dest": "LAX"},                                     # SFO-LAX direct 337 mi, via DEN 1830 mi: detour > 1.6
    {"dep_wall": pd.Timestamp("2026-03-02 10:20"), "flight_date": pd.Timestamp("2026-03-02")},   # layover 20 < M
    {"dep_wall": pd.Timestamp("2026-03-02 14:10"), "flight_date": pd.Timestamp("2026-03-02")},   # layover 250 > 240
    {"flight_date": pd.Timestamp("2026-03-03"), "dep_wall": pd.Timestamp("2026-03-03 11:00")},   # other day
    {"fn_marketing": 1},                                 # through flight (same marketing number)
    {"carrier": "UA", "fn_operating": 1},                # through flight (same operating number)
    {"tail": "N1"},                                      # same aircraft: rotation, not a connection
])
def test_ineligible_pairs_are_excluded(b_override):
    a = {**A, "tail": "N1"}
    assert len(build(frame(a, {**B, **b_override}))) == 0
    assert len(build(frame(A, B))) == 1                  # control


def test_as_and_ha_are_one_network_and_same_airline_family_only():
    a = {**A, "network": "AS", "carrier": "AS"}
    b = {**B, "network": "HA", "carrier": "QX"}
    assert len(build(frame(a, b))) == 1


def test_cross_midnight_leg2_next_day_pairs_on_arrival_date():
    a = leg("SFO", "DEN", "2026-03-02 21:30", "2026-03-03 00:30", fn=1)
    b = leg("DEN", "ORD", "2026-03-03 01:30", "2026-03-03 04:30", fn=2)
    r = build(frame(a, b))
    assert len(r) == 1 and r.layover[0] == 60 and r.arr_date[0] == pd.Timestamp("2026-03-03")


def test_one_pair_per_leg1_and_bucket_with_weights():
    legs = [A]
    for k, hh in enumerate(["10:50", "11:00", "11:10", "12:30", "13:10"]):   # layovers 50, 60, 70, 150, 190
        legs.append(leg("DEN", "ORD", f"2026-03-02 {hh}", f"2026-03-02 15:00", fn=10 + k))
    r = build(frame(*legs))
    # edges for M=40: [40,70) [70,100) [100,120) [120,241): buckets 0,0,1,3,3 -> three picks
    assert sorted(r.bucket.tolist()) == [0, 1, 3]
    w = dict(zip(r.bucket, r.w))
    assert w[0] == 2 and w[1] == 1 and w[3] == 2
    assert build(frame(*legs), seed=42).equals(build(frame(*legs), seed=42))     # deterministic


def test_rows_after_protocol_window_are_refused():
    late = leg("SFO", "DEN", "2026-09-02 07:00", "2026-09-02 10:00")
    with pytest.raises(ValueError):
        build(frame(late, B))


def test_outcomes_unit():
    miss, dis = pairs.outcomes(np.array([30.0]), np.array([0.0]), np.array([False]), np.array([False]),
                               np.array([False]), np.array([60]), np.array([40]))
    assert miss[0] and dis[0]
