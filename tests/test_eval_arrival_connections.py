"""Helpers and guards of the evaluation script (no data files)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
sys.path.insert(0, str(ROOT))
import eval_arrival_connections as E  # noqa: E402

RNG = np.random.default_rng(5)


def test_window_guard_refuses_2026_09_and_before_2025(monkeypatch):
    E.assert_window(["2025-01", "2026-08"])
    for bad in (["2026-09"], ["2026-08", "2027-01"], ["2024-12"]):
        with pytest.raises(ValueError):
            E.assert_window(bad)
    monkeypatch.setenv("FLYRELY_HOLDOUT_UNLOCK", "2026-09")
    with pytest.raises(SystemExit):
        E.assert_window(["2026-01"])


def test_configs_never_touch_2026_09():
    for cfg in (E.FULL, E.QUICK):
        months = set()
        for k in ("val", "select", "confirm", "rho_fit", "rho_sel"):
            months |= set(E.months_between(*cfg[k]))
        E.assert_window(sorted(months))
    assert max(E.months_between(*E.FULL["val"])) == "2026-08"
    assert max(E.months_between(*E.QUICK["val"])) < "2026-01"          # quick reads 2025 only


def test_store_refuses_locked_month():
    s = E.Store(None, [], None, "x")
    with pytest.raises(ValueError):
        s.meta("2026-09")
    with pytest.raises(ValueError):
        s.pred("2026-09")


def test_day_block_ci_covers_the_mean_and_is_deterministic():
    day = np.repeat(np.arange(100), 50)
    v = RNG.normal(0.3, 1.0, len(day))
    ci = E.day_block_ci(day, v)
    assert ci["lo"] < v.mean() < ci["hi"] and ci == E.day_block_ci(day, v)
    assert E.day_block_ci(day, v - 5)["hi"] < 0
    sk = E.day_block_ci(day, np.ones(len(day)), np.full(len(day), 2.0), kind="skill")
    assert sk["point"] == pytest.approx(0.5)


def test_metrics_and_deciles():
    p = RNG.uniform(0, 1, 20000)
    y = (RNG.random(20000) < p).astype(float)
    m = E.binary_metrics(y, p)
    assert m["ece"] < 0.03 and 0.6 < m["auc"] < 0.9 and len(m["deciles"]) == 10
    biased = E.binary_metrics(y, np.clip(p + 0.1, 0, 1))
    assert biased["ece"] > 0.07
    w = np.ones(len(y))
    assert E.weighted_ece(y, p, w) == pytest.approx(m["ece"], abs=2e-3)


def test_ordered_rates():
    assert E.ordered_rates([0.1, 0.2, 0.4]) and E.ordered_rates([0.1, None, 0.4])
    assert not E.ordered_rates([0.1, 0.3, 0.2]) and not E.ordered_rates([0.2, 0.2, 0.2])
    assert E.ordered_rates([0.1, 0.1, 0.4]) and not E.ordered_rates([0.1, 0.1, 0.4], strict=True)
    assert list(E.band_of(np.array([0.01, 0.05, 0.149, 0.15, 0.9]))) == [0, 1, 1, 2, 2]
    assert list(E.tier_of(np.array([0.1, 0.2, 0.299, 0.3]))) == [0, 1, 1, 2]


def test_base_rate_table_lookup_and_fallbacks():
    b = {"hub": np.array(["DEN"] * 4), "layover": np.array([50, 52, 53, 100]), "miss": np.array([1, 0, 0, 0])}
    tab = E.base_rate_table(b)
    out = E.bl_lookup(tab, np.array(["DEN", "DEN", "ATL"]), np.array([51, 101, 51]))
    assert out[0] == pytest.approx(1 / 3)        # DEN, cell 10: layovers 50, 52, 53 -> 1 of 3
    assert out[1] == 0.0
    assert out[2] == pytest.approx(1 / 3)        # unseen hub falls back to the layover bucket's rate in that cell


def test_run_is_refused_when_unlock_variable_set(monkeypatch):
    monkeypatch.setenv("FLYRELY_HOLDOUT_UNLOCK", "2026-09")
    with pytest.raises(SystemExit):
        E.run(E.QUICK)


def test_arrival_gate_withholds_copula_numbers_when_ladder_not_shipped():
    def mk(a_out, c_out):
        return ({"decisions": {"outcome": a_out}},
                {"decisions": {"outcome": c_out}, "_final_dm": {"decision_outcome": c_out}})
    for a_out in E.LADDER_USABLE_FOR_CONNECTIONS:
        a, c = mk(a_out, E.COPULA_SHIP)
        assert E.apply_arrival_gate(a, c) == E.COPULA_SHIP and c["_final_dm"]["decision_outcome"] == E.COPULA_SHIP
    for a_out in ("ship_nothing_arrival", "ship_nothing_arrival_other"):
        a, c = mk(a_out, E.COPULA_SHIP)
        assert E.apply_arrival_gate(a, c) == "no_connection_numbers_arrival_ladder_not_shipped"
        assert c["_final_dm"]["decision_outcome"] != E.COPULA_SHIP
        a, c = mk(a_out, "serve_layover_table_B-L_itin-layover-table-v1")       # B-L uses no ladder
        assert E.apply_arrival_gate(a, c) == "serve_layover_table_B-L_itin-layover-table-v1"


def test_connection_metrics_score_miss_on_operated_pairs_only():
    n = 4000
    rng = np.random.default_rng(1)
    pm = rng.uniform(0.01, 0.4, n)
    op = rng.random(n) > 0.2                                    # 20% of pairs have a cancelled leg
    miss = op & (rng.random(n) < pm)                            # calibrated given both legs operate
    ev = {"miss": miss, "disrupted": miss | ~op, "op": op, "w": np.ones(n), "layover": rng.integers(40, 240, n),
          "bucket": rng.integers(0, 4, n), "hub": np.array(["DEN", "ATL"] * (n // 2)), "ym": np.full(n, "2025-11"),
          "same_carrier": np.ones(n, bool), "c1p": np.full(n, 0.1), "c2p": np.full(n, 0.1),
          "day": np.repeat(np.arange(40), n // 40), "p": {"R1": pm}}
    res = E.connection_metrics(ev, pm, pm, pm)
    assert res["n_pairs"] == int(op.sum()) and res["n_pairs_all"] == n
    assert res["observed_miss_rate"] == pytest.approx(miss[op].mean())
    assert res["disrupted"]["n"] == n
