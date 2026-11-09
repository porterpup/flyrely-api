"""Ladder maths and the training artifact (tiny synthetic fit, no data files)."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
sys.path.insert(0, str(ROOT))
from flyrely_dist import ladder as L  # noqa: E402

RNG = np.random.default_rng(0)


def fake_delays(n=50000):
    d = np.where(RNG.random(n) < 0.25, RNG.exponential(45, n) + 10, RNG.normal(-8, 12, n))
    d[RNG.random(n) < 0.003] = np.inf
    return d


TAIL = L.build_tail_table(fake_delays(), L.ARR_THRESHOLDS[0], L.ARR_THRESHOLDS[-1])


def ladder(probs=None):
    p = probs if probs is not None else [0.95, 0.85, 0.6, 0.23, 0.16, 0.12, 0.09, 0.06, 0.04, 0.02, 0.008]
    return L.Ladder.from_probs(L.ARR_THRESHOLDS, p, TAIL)


def test_rearrangement_makes_sf_monotone_and_bounded():
    bad = [0.9, 0.95, 0.6, 0.62, 0.2, 0.3, 0.1, 0.05, 0.06, 0.01, 0.02]
    assert L.needs_rearrangement(np.array([bad]))[0]
    lad = ladder(bad)
    x = np.arange(-80, 700)
    s = lad.sf(x)
    assert (np.diff(s) <= 1e-12).all() and s.min() >= 0 and s.max() <= 1
    assert not L.needs_rearrangement(np.array([lad.probs])).any()


def test_interpolation_exact_at_thresholds_and_linear_between():
    lad = ladder()
    for t, p in zip(lad.thresholds, lad.probs):
        assert lad.sf(t) == pytest.approx(p, abs=1e-12)
    assert lad.sf(22.5) == pytest.approx((lad.probs[3] + lad.probs[4]) / 2)


def test_severity_sums_to_one_and_matches_threshold_points():
    lad = ladder()
    s = lad.severity()
    assert sum(s.values()) == pytest.approx(1.0)
    p = dict(zip(lad.thresholds, lad.probs))
    assert s["minor"] == pytest.approx((p[15] - p[45]) / p[15])
    assert s["moderate"] == pytest.approx((p[45] - p[120]) / p[15])
    assert s["severe"] == pytest.approx(p[120] / p[15])
    z = L.severity_matrix(np.zeros((1, 11)), L.ARR_THRESHOLDS)[0]
    assert list(z) == [1.0, 0.0, 0.0]


def test_quantile_inverts_sf():
    lad = ladder()
    for q in (0.5, 0.8, 0.9):
        x = lad.quantile(q)
        assert lad.sf(x) == pytest.approx(1 - q, abs=2e-3)
    assert lad.quantile(0.5) < lad.quantile(0.8) < lad.quantile(0.9)


def test_tails_stay_in_unit_interval_and_cap():
    lad = ladder()
    x = np.array([-500, -60, -31, 301, 600, 5000])
    s = lad.sf(x)
    assert ((s >= 0) & (s <= 1)).all()
    assert s[-1] == s[-2]                              # clamped at the 600-minute cap
    assert TAIL.hi_sf[0] == 1.0 and TAIL.lo_cdf[-1] == 1.0
    assert (np.diff(TAIL.hi_sf) <= 0).all() and (np.diff(TAIL.lo_cdf) >= 0).all()


def test_diverted_mass_survives_to_the_cap():
    d = np.array([10.0] * 90 + [np.inf] * 10 + [-50.0] * 100)
    t = L.build_tail_table(d, -30, 300)
    assert t.hi_sf[-1] == pytest.approx(1.0)           # of those >= 300 (all diverted) all remain at the cap


def test_batch_matches_single():
    P = L.rearrange(np.array([[0.9, 0.8, 0.5, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.01, 0.005],
                              [0.97, 0.9, 0.7, 0.35, 0.25, 0.2, 0.15, 0.1, 0.06, 0.03, 0.01]]))
    S = L.curve_matrix(P, L.ARR_THRESHOLDS, TAIL)
    for i in range(2):
        one = L.Ladder(L.ARR_THRESHOLDS, tuple(P[i]), TAIL)
        assert np.allclose(one.sf(L.GRID), S[i])
        assert one.quantile(0.9) == pytest.approx(L.quantile_matrix(S, 0.9)[i])


def test_logistic_recal_recovers_a_shift():
    p = RNG.uniform(0.05, 0.6, 40000)
    true = 1 / (1 + np.exp(-(0.4 + 1.0 * np.log(p / (1 - p)))))
    y = (RNG.random(len(p)) < true).astype(int)
    a, b = L.fit_logistic_recal(p, y)
    assert a == pytest.approx(0.4, abs=0.08) and b == pytest.approx(1.0, abs=0.1)
    assert abs(L.apply_logistic_recal(p, a, b).mean() - y.mean()) < 0.01


def test_bundle_trains_saves_loads_and_checks_feature_names(tmp_path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    import train_ladders as tl

    names = list(joblib.load(ROOT / "models" / "feature_names.joblib"))
    n = 3000
    X = RNG.random((n, len(names))).astype(np.float32)
    arr = np.where(X[:, 0] > 0.7, RNG.exponential(40, n) + 10, RNG.normal(-8, 10, n))
    lab = {"arr_delay": arr, "dep_delay": arr + 3, "cancelled": RNG.random(n) < 0.02}
    lab["arr_delay"][lab["cancelled"]] = np.nan
    lab["dep_delay"][lab["cancelled"]] = np.nan

    def fitter(Xf, yf, Xv, yv):
        m = HistGradientBoostingClassifier(max_iter=5, random_state=42).fit(Xf, yf)
        return m, IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(Xv)[:, 1], yv)

    bundle = tl.train_bundle(X, lab, X, lab, names, fitter, log=lambda s: None)
    assert len(bundle.arr) == 11 and len(bundle.dep) == 10 and bundle.cancel.threshold is None
    path = tmp_path / "b.joblib"
    bundle.save(path)
    mp = tl.write_manifest(path)
    assert "sha256" in mp.read_text()
    b2 = L.load_bundle(path)
    r = b2.predict_arrays(X[:5])
    assert r["arr"].shape == (5, 11) and r["dep"].shape == (5, 10) and r["cancel"].shape == (5,)
    assert (np.diff(r["arr"], axis=1) <= 1e-12).all()
    lad = b2.ladders(X[:1])[0][0]
    assert 0 <= lad.prob_late(15) <= 1
    bundle.feature_names = names[::-1]
    bundle.save(path)
    with pytest.raises(ValueError):
        L.load_bundle(path)
