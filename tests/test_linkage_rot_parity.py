"""Code parity of linkage.rot_live with the frozen flyrely_features.rotation (L1 plan section 2.5, task T4)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
from flyrely_features.rotation import Rotation, rotation_features  # noqa: E402
from linkage.rot_live import ROT_COLS, rot_live, true_chain  # noqa: E402

N_ROWS = 2000
NAN = np.nan
# features that depend on a leg's cancelled flag; the others must never move with strict_asof
CANCEL_DEPENDENT = ["rot_known_delay", "rot_leg_pending", "rot_leg_cancelled", "rot_projected_delay"]
CANCEL_FREE = [k for k in ROT_COLS if k not in CANCEL_DEPENDENT]


def synthetic(seed=7, n_tails=160, legs=14):
    """Many tails with chains of legs over about 3 days: ties in departure minute, cancellations, missing tails."""
    rng = np.random.default_rng(seed)
    tail, s_dep, elapsed, dly, canc = [], [], [], [], []
    for t in range(n_tails):
        x = rng.integers(0, 600)
        for _ in range(legs):
            x += rng.choice([0, 40, 90, 150, 240, 400, 900])      # 0: a tie with the previous leg's minute
            tail.append(t)
            s_dep.append(float(x))
            elapsed.append(float(rng.integers(45, 300)))
            dly.append(float(rng.integers(-15, 200)))
            canc.append(rng.random() < 0.08)
    n = len(tail)
    tail = np.array(tail)
    tail[rng.random(n) < 0.02] = -1                              # no tail recorded
    s_dep = np.array(s_dep) + 1.0e4
    s_dep[rng.random(n) < 0.01] = NAN                            # no scheduled time
    el = np.array(elapsed)
    el[rng.random(n) < 0.01] = NAN
    return finish(dict(tail_id=tail, s_dep=s_dep, elapsed=np.nan_to_num(el), s_arr=s_dep + el,
                       dep_delay=np.array(dly), cancelled=np.array(canc)))


def finish(A):
    A["a_dep"] = np.where(A["cancelled"], NAN, A["s_dep"] + np.nan_to_num(A["dep_delay"]))
    return A


def assert_same(a, b, keys=ROT_COLS):
    for k in keys:
        np.testing.assert_allclose(a[k], b[k], atol=1e-9, rtol=0, equal_nan=True, err_msg=k)


def frozen_and_live(A, rows, c, **kw):
    R = Rotation(A)
    return rotation_features(A, R, rows, c), rot_live(A, rows, c, true_chain(A, R, rows, c), **kw)


def sample_rows(A, n=N_ROWS, seed=1):
    ok = np.flatnonzero((A["tail_id"] >= 0) & ~np.isnan(A["s_dep"]))
    return np.sort(np.random.default_rng(seed).choice(ok, size=min(n, len(ok)), replace=False))


@pytest.mark.parametrize("minutes", [180, 60, 1000, 0])
def test_synthetic_true_chain_equals_frozen_rotation(minutes):
    A = synthetic()
    rows = sample_rows(A)
    c = A["s_dep"][rows] - minutes + np.where(np.arange(len(rows)) % 3 == 0, 0.5, 0.0)   # fractional cutoffs too
    frozen, live = frozen_and_live(A, rows, c, strict_asof=False)
    assert len(rows) == N_ROWS
    assert_same(frozen, live)
    if minutes > 0:                                                # (at 0 the cutoff is B's own departure: no leg)
        assert frozen["rot_has_leg"].sum() > 500                  # the fixture actually exercises the features
        assert np.nansum(frozen["rot_leg_cancelled"]) > 5 and np.nansum(frozen["rot_legs_between"] > 0) > 50


def test_synthetic_all_rows_including_untailed_and_late_cutoffs():
    A = synthetic(seed=11)
    rows = np.arange(len(A["s_dep"]))
    c = np.nan_to_num(A["s_dep"]) - 120
    c[::7] += 500                                                  # cutoff after the flight itself: no leg
    frozen, live = frozen_and_live(A, rows, c)
    assert_same(frozen, live)


def test_strict_asof_changes_only_cancelled_legs():
    A = synthetic()
    rows = sample_rows(A)
    c = A["s_dep"][rows] - 180
    R = Rotation(A)
    frozen = rotation_features(A, R, rows, c)
    chain = true_chain(A, R, rows, c)
    # default: a cancellation is visible once the leg's departure time has passed; every chosen leg is past it
    assert_same(frozen, rot_live(A, rows, c, chain, strict_asof=True))
    # cancellation received 90 min after the scheduled departure: some are still unknown at c
    recv = A["s_dep"] + 90
    strict = rot_live(A, rows, c, chain, strict_asof=True, cancel_recv=recv)
    assert_same(frozen, strict, CANCEL_FREE)
    changed = np.zeros(len(rows), bool)
    for k in CANCEL_DEPENDENT:
        changed |= ~np.isclose(frozen[k], strict[k], atol=1e-9, rtol=0, equal_nan=True)
    cancelled = frozen["rot_leg_cancelled"] == 1
    assert changed.sum() > 0
    assert changed[~cancelled].sum() == 0                          # untouched wherever the leg was not cancelled
    unseen = cancelled & (recv[chain[:, -1]] > c) | cancelled      # cancelled rows are the only candidates to change
    assert changed[unseen].sum() == changed.sum()
    # an unseen cancellation looks like a pending leg with an unknown delay at c
    d = changed & (strict["rot_leg_cancelled"] == 0)
    assert np.all(strict["rot_leg_pending"][d] == 1) and not np.isnan(strict["rot_known_delay"][d]).any()


def test_strict_asof_cancellation_visible_only_after_receipt():
    A = finish(dict(tail_id=np.array([1, 1]), s_dep=np.array([1000.0, 1300.0]), s_arr=np.array([1120.0, 1400.0]),
                    elapsed=np.array([120.0, 100.0]), dep_delay=np.array([NAN, 0.0]),
                    cancelled=np.array([True, False])))
    chain = np.array([[0]])
    rows = np.array([1])
    recv = np.array([1100.0, 0.0])
    early = rot_live(A, rows, np.array([1050.0]), chain, strict_asof=True, cancel_recv=recv)
    late = rot_live(A, rows, np.array([1150.0]), chain, strict_asof=True, cancel_recv=recv)
    assert early["rot_leg_cancelled"][0] == 0 and early["rot_leg_pending"][0] == 1 and early["rot_known_delay"][0] == 50
    assert late["rot_leg_cancelled"][0] == 1 and np.isnan(late["rot_known_delay"][0])
    hind = rot_live(A, rows, np.array([1050.0]), chain, strict_asof=False)   # hindsight flag sees it at once
    assert hind["rot_leg_cancelled"][0] == 1


def test_explicit_chain_with_a_hole_is_not_a_chain():
    A = finish(dict(tail_id=np.array([1, 1, 1]), s_dep=np.array([1000.0, 1200.0, 1500.0]),
                    s_arr=np.array([1100.0, 1300.0, 1600.0]), elapsed=np.array([100.0] * 3),
                    dep_delay=np.zeros(3), cancelled=np.zeros(3, bool)))
    rows, c = np.array([2]), np.array([1250.0])
    ok = rot_live(A, rows, c, np.array([[0, 1]]))
    assert ok["rot_has_leg"][0] == 1 and ok["rot_legs_between"][0] == 0
    hole = rot_live(A, rows, c, np.array([[1, -1]]))
    assert hole["rot_has_leg"][0] == 0


def bts_months():
    d = ROOT / "data" / "bts" / "parquet_v2"
    return d if d.exists() and (d / "2025-06.parquet").exists() else None


@pytest.mark.skipif(bts_months() is None, reason="data/bts not present")
@pytest.mark.parametrize("minutes", [180, 60])
def test_bts_sample_true_chain_equals_frozen_rotation(minutes):
    import l1_bts_calib
    df, A = l1_bts_calib.load_window("2025-06", "2025-06")
    assert df["FlightDate"].max() < np.datetime64("2026-09-01")
    rows = sample_rows(A, seed=3)
    c = A["s_dep"][rows] - minutes
    frozen, live = frozen_and_live(A, rows, c, strict_asof=False)
    assert len(rows) == N_ROWS
    assert_same(frozen, live)
    assert frozen["rot_has_leg"].mean() > 0.8
    strict = rot_live(A, rows, c, true_chain(A, Rotation(A), rows, c), strict_asof=True,
                      cancel_recv=A["s_dep"] + 120)
    assert_same(frozen, strict, CANCEL_FREE)
    other = frozen["rot_leg_cancelled"] != 1
    for k in CANCEL_DEPENDENT:
        np.testing.assert_allclose(frozen[k][other], strict[k][other], atol=1e-9, rtol=0, equal_nan=True)
