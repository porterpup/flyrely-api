"""Pure helpers in backtest/opensky_coverage.py: callsign mapping, N-number -> icao24, matching math."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
import opensky_client as oc  # noqa: E402
import opensky_coverage as oco  # noqa: E402


def test_n_number_known_values():
    assert oco.n_number_to_icao24("N1") == "a00001"
    assert oco.n_number_to_icao24("N99999") == "adf7c7"   # last US block address
    assert oco.n_number_to_icao24("N12345") == "a061d9"
    assert oco.n_number_to_icao24("N1A") == "a00002"
    assert oco.n_number_to_icao24("n542de") == "a6dfaa"   # case-insensitive


def test_n_number_invalid():
    for bad in (None, "", "C-FABC", "N0", "N1I", "N1O", "N123456", "N1A2", "XYZ"):
        assert oco.n_number_to_icao24(bad) is None


def test_callsign_candidates():
    assert oco.callsign_candidates("9E", 5123, "DL", 5123) == ["EDV5123", "DAL5123"]
    assert oco.callsign_candidates("DL", 1209, "DL", 1209) == ["DAL1209"]
    assert oco.callsign_candidates("ZZ", 1, "UA", 77) == ["UAL77"]
    assert oco.callsign_candidates("AA", None) == []
    assert all(oco.IATA_TO_ICAO[c] for c in oco.REGIONALS)


def test_norm_callsign():
    assert oco.norm_callsign("DAL1209 ") == "DAL1209"
    assert oco.norm_callsign("   ") is None
    assert oco.norm_callsign(None) is None


def test_stats_helpers():
    s = oco.abs_stats([-10, 0, 10, 20])
    assert s["n"] == 4 and s["bias_mean_min"] == 5 and s["median_abs_min"] == 10
    p = oco.prf([1, 1, 0, 0], [1, 0, 1, 0])
    assert p["precision"] == 0.5 and p["recall"] == 0.5


def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(oc.urllib.request, "urlopen", boom)


def _ledger(tmp_path, monkeypatch, rows):
    monkeypatch.setattr(oc, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(oc, "LEDGER", tmp_path / "ledger.jsonl")
    (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_budget_floor_refuses_before_network(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    _ledger(tmp_path, monkeypatch, [{"group": "/flights", "status": 200, "remaining": 1510, "cost_observed": 30}])
    with pytest.raises(oc.BudgetStop):   # 1510-30 < 1500 floor
        oc.OpenSky().get("/flights/all", {"begin": 1, "end": 2}, 30)


def test_total_cap_refuses_before_network(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    _ledger(tmp_path, monkeypatch, [{"group": "/flights", "status": 200, "remaining": 3000, "cost_observed": 2490}])
    with pytest.raises(oc.BudgetStop):   # 2490 + 30 > 2500 experiment cap
        oc.OpenSky().get("/flights/all", {"begin": 1, "end": 2}, 30)


def test_group_cap_is_per_group(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    _ledger(tmp_path, monkeypatch, [{"group": "/flights", "status": 200, "remaining": 1750, "cost_observed": 2250},
                                    {"group": "/states", "status": 200, "remaining": 3000, "cost_observed": 598}])
    with pytest.raises(oc.BudgetStop):   # /states spent 598 + 4 > 600 cap
        oc.OpenSky().get("/states/all", {"lamin": 24}, 4, cache_tag="t", group_cap=600)


def test_cache_hit_costs_nothing(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    _ledger(tmp_path, monkeypatch, [])
    key, _ = oc.cache_key("/flights/all", {"begin": 1, "end": 2})
    (tmp_path / f"{key}.json").write_text('{"body": [1]}')
    assert oc.OpenSky().get("/flights/all", {"begin": 1, "end": 2}, 30) == ([1], True)
    k2, _ = oc.cache_key("/states/all", {"a": 1}, "poll-1")
    assert k2 != oc.cache_key("/states/all", {"a": 1}, "poll-2")[0]   # snapshots with different tags do not collide


def test_haversine():
    assert abs(oco.haversine_km(33.6367, -84.4281, 33.6367, -84.4281)) < 1e-6
    assert 1200 < float(oco.haversine_km(33.64, -84.43, 40.64, -73.78)) < 1250   # ATL-JFK ~1,222 km
