"""Connection engine: copula miss probability, N-leg product, MCT, dependence fitting (no data files)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flyrely_dist import connections as C  # noqa: E402
from flyrely_dist import ladder as L  # noqa: E402
from flyrely_dist import mct  # noqa: E402

RNG = np.random.default_rng(3)
D = RNG.normal(-8, 14, 40000)
D = np.where(RNG.random(40000) < 0.2, RNG.exponential(50, 40000) + 12, D)
D[RNG.random(40000) < 0.003] = np.inf
TAIL_A = L.build_tail_table(D, -30, 300)
TAIL_D = L.build_tail_table(D, -5, 300)
A1 = [0.95, 0.85, 0.6, 0.23, 0.16, 0.12, 0.09, 0.06, 0.04, 0.02, 0.008]
D1 = [0.7, 0.4, 0.2, 0.13, 0.1, 0.075, 0.05, 0.03, 0.015, 0.006]


def arr(p=A1):
    return L.Ladder.from_probs(L.ARR_THRESHOLDS, p, TAIL_A)


def dep(p=D1):
    return L.Ladder.from_probs(L.DEP_THRESHOLDS, p, TAIL_D)


def fc(a=None, d=None, c=0.01):
    return C.LegForecast(a or arr(), d or dep(), c)


class Step:
    """Point mass at `at` (minutes) with `beyond` of the mass beyond the cap."""
    def __init__(self, at, beyond=0.0):
        self.at, self.beyond = at, beyond

    def sf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x <= self.at, 1.0, self.beyond)


def test_point_masses_give_zero_or_one():
    for xd, yd, s in [(30, 0, 20), (30, 0, 30), (30, 0, 29), (-10, 0, -5), (50, 40, 5), (50, 40, 10)]:
        p = C._miss(Step(xd), Step(yd), s, 0.0)
        assert p == (1.0 if xd - yd > s else 0.0), (xd, yd, s, p)


def test_independence_matches_brute_force_convolution():
    a, d = arr(), dep()
    x = L.GRID.astype(float)
    sa, sd = a.sf(x), d.sf(x)
    pa = np.clip(sa - np.append(sa[1:], sa[-1]), 0, None)
    pd_ = np.clip(sd - np.append(sd[1:], sd[-1]), 0, None)
    for s in (-5, 0, 10, 20, 55):
        brute = float(sa[-1])
        for xi, px in zip(L.GRID, pa):
            brute += px * pd_[L.GRID < xi - s].sum()
        got = C.miss_probability(fc(a, d), fc(a, d), 40 + s, 40, C.DependenceModel("R0"))
        assert abs(got - brute) < 1e-3, (s, got, brute)


def test_miss_probability_falls_with_layover_and_rises_with_mct():
    f1, f2 = fc(), fc()
    dm = C.DependenceModel("R1", 0.3)
    by_layover = [C.miss_probability(f1, f2, L_, 40, dm) for L_ in (40, 50, 70, 100, 150, 240)]
    assert all(a >= b - 1e-12 for a, b in zip(by_layover, by_layover[1:])) and by_layover[0] > by_layover[-1]
    by_mct = [C.miss_probability(f1, f2, 90, m, dm) for m in (30, 40, 50, 60)]
    assert all(a <= b + 1e-12 for a, b in zip(by_mct, by_mct[1:])) and by_mct[0] < by_mct[-1]


def test_later_departure_of_leg2_rescues():
    f1 = fc()
    early = fc(d=dep([0.5, 0.2, 0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]))
    late = fc(d=dep([0.99, 0.95, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.03]))
    assert C.miss_probability(f1, late, 60, 40) < C.miss_probability(f1, early, 60, 40)


def test_perfect_correlation_with_identical_marginals():
    class Same:
        def sf(self, v):
            return arr().sf(v)
    tail = float(arr().sf(600))                     # mass beyond the cap always counts as a miss
    ok = C._miss(Same(), Same(), 10.0, 0.9999)
    bad = C._miss(Same(), Same(), -5.0, 0.9999)
    assert ok == pytest.approx(tail, abs=0.002)     # X = Y, s >= 0: P(0 > s) = 0 plus the tail mass
    assert bad > 0.99                               # s < 0: always missed (0 > s)
    assert C._miss(Same(), Same(), 10.0, 0.0) > ok     # independence is worse than comonotone here


def test_diverted_and_beyond_cap_mass_is_a_miss():
    p = C._miss(Step(0, beyond=0.2), Step(-20), 100, 0.0)
    assert p == pytest.approx(0.2, abs=1e-9)
    p = C._miss(Step(0, beyond=0.2), Step(-20), 100, 0.5)
    assert p == pytest.approx(0.2, abs=1e-6)


def test_three_legs_are_flagged_approximate_and_two_are_not():
    fs = [fc(), fc(), fc()]
    s = C.trip_summary(fs, [0.1, 0.05])
    assert s["approximate"] is True
    assert s["disruption_probability"] == pytest.approx(1 - 0.99 ** 3 * 0.9 * 0.95)
    s2 = C.trip_summary(fs[:2], [0.1])
    assert s2["approximate"] is False and s2["disruption_probability"] == pytest.approx(1 - 0.99 ** 2 * 0.9)
    assert C.trip_summary(fs[:2], [None])["disruption_probability"] is None
    with pytest.raises(ValueError):
        C.trip_summary(fs, [0.1])


def test_sf_at_equals_ladder_sf_and_pit_is_uniform():
    P = L.rearrange(np.array([A1, A1]))
    for x in (-80, -45, -30, -7, 0, 14, 15, 46, 299, 300, 450, 700):
        got = C.sf_at(P, L.ARR_THRESHOLDS, TAIL_A, np.array([x, x]))
        assert got[0] == pytest.approx(float(arr().sf(x)), abs=1e-9)
    # PIT of draws from the ladder's own distribution is roughly uniform
    n = 20000
    Pn = np.tile(P[0], (n, 1))
    u = RNG.random(n)
    grid_s = arr().sf(L.GRID.astype(float))
    x = np.array([L.GRID[np.argmax(grid_s <= 1 - v)] for v in u])
    z = C.pit_normal_scores(Pn, L.ARR_THRESHOLDS, TAIL_A, x, np.random.default_rng(1))
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05


def test_fit_dependence_recovers_rho_and_r2_shrinks():
    n = 30000
    z1 = RNG.normal(size=n)
    z2 = 0.4 * z1 + np.sqrt(1 - 0.16) * RNG.normal(size=n)
    hubs = np.array(["DEN"] * n)
    lay = np.full(n, 50)
    g = C.fit_dependence(z1, z2, hubs, lay, variant="R1")
    assert g.global_rho == pytest.approx(0.4, abs=0.02)
    r2 = C.fit_dependence(z1, z2, hubs, lay, variant="R2")
    assert r2.rho("DEN", 50) == pytest.approx(0.4, abs=0.03)
    assert r2.rho("ATL", 50) == r2.global_rho                     # unseen cell falls back to the global value
    small = C.DependenceModel("R2", 0.2, {"DEN|lt60": (0.8, 50)}, 5000.0)
    assert 0.2 < small.rho("DEN", 30) < 0.25                      # n=50 barely moves it
    assert C.DependenceModel("R0").rho("DEN", 30) == 0.0


def test_resolve_mct_defaults_override_and_cross_network():
    assert mct.resolve_mct("DEN") == (50, "default_large_campus")
    assert mct.resolve_mct("DEN", user_minutes=35) == (35, "user")
    assert mct.resolve_mct("MCO") == (40, "default_top30")
    assert mct.resolve_mct("XYZ") == (30, "default_small")
    assert mct.resolve_mct("BUF", deps_per_day=120) == (35, "default_mid")
    assert mct.resolve_mct("MCO", cross_network=True) == (70, "default_top30+cross_network")
    table = mct.load_table(ROOT / "models" / "mct_defaults.csv")
    assert mct.resolve_mct("ORD", table)[0] == 50 and mct.resolve_mct("PDX", table)[0] == 40


def test_provider_builds_forecasts_from_a_bundle():
    from datetime import datetime
    from sklearn.dummy import DummyClassifier
    from sklearn.isotonic import IsotonicRegression
    from flyrely_dist.providers import BookingLadderProvider, LegSpec
    names = ["a", "b"]
    X = RNG.random((200, 2))
    y = (RNG.random(200) < 0.3).astype(int)
    dm = DummyClassifier(strategy="prior").fit(X, y)
    iso = IsotonicRegression(out_of_bounds="clip").fit(dm.predict_proba(X)[:, 1] + RNG.random(200) * 1e-3, y)
    tm = lambda t: L.ThresholdModel(t, dm, iso)
    bundle = L.LadderBundle("t", names, [tm(t) for t in L.ARR_THRESHOLDS], [tm(t) for t in L.DEP_THRESHOLDS],
                            tm(None), TAIL_A, TAIL_D, "2025-12")
    prov = BookingLadderProvider(bundle, feature_fn=lambda o, d, t, a: {"a": 0.1, "b": 0.2})
    out = prov.forecast([LegSpec("SFO", "DEN", datetime(2026, 12, 3, 7, 10), "UA")] * 2)
    assert len(out) == 2 and out[0].provider == "booking-ladder-v1" and 0 <= out[0].p_cancel <= 1
    assert 0 <= C.miss_probability(out[0], out[1], 60, 40) <= 1
