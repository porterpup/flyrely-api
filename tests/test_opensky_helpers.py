"""Pure helpers in backtest/opensky_coverage.py: callsign mapping, N-number -> icao24, matching math."""

import sys
from pathlib import Path

import numpy as np

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


def test_budget_guard_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(oc, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(oc, "LEDGER", tmp_path / "ledger.jsonl")
    (tmp_path / "ledger.jsonl").write_text('{"group": "/flights", "status": 200, "remaining": 1510, "cost_observed": 30}\n')
    c = oc.OpenSky()
    try:
        c.get("/flights/all", {"begin": 1, "end": 2}, 30)  # 1510-30 < 1500 floor -> must refuse before any network call
        assert False, "expected BudgetStop"
    except oc.BudgetStop:
        pass
    key, _ = oc.cache_key("/flights/all", {"begin": 1, "end": 2})
    (tmp_path / f"{key}.json").write_text('{"body": [1]}')
    assert c.get("/flights/all", {"begin": 1, "end": 2}, 30) == ([1], True)  # cache hit costs nothing
