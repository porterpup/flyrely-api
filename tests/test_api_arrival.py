"""T6: optional arrival_time on /predict, the nullable `arrival` block, /health, flags and the ladder manifest."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import arrival_fakes
import arrival_service
from flyrely_dist.providers import LegSpec
import main

ROOT = Path(__file__).resolve().parent.parent
SNAP = json.loads((ROOT / "tests" / "fixtures" / "predict_snapshot.json").read_text())
client = TestClient(main.app)
REAL_LADDER = Path(os.getenv("FLYRELY_TEST_LADDER_PATH", ROOT / "models" / "arrival_ladder_v1.joblib"))


@pytest.fixture
def arrival_on(monkeypatch):
    monkeypatch.setattr(main, "ARRIVAL_FLAG", True)
    seen = []
    monkeypatch.setattr(main, "arrival_service", arrival_fakes.make_service(seen))
    return seen


@pytest.mark.parametrize("name", ["plain", "codeshare"])
def test_old_predict_output_is_unchanged_with_flags_off(name):
    """Fixed requests answered by origin/master (before T6): same keys, same values."""
    assert not main.ARRIVAL_FLAG
    r = client.post("/predict", json=SNAP[name]["request"])
    assert r.status_code == SNAP[name]["status"]
    assert r.json() == SNAP[name]["body"]


def test_arrival_time_is_echoed_and_does_not_change_the_prediction():
    req = {**SNAP["plain"]["request"], "arrival_time": "2030-11-15T21:55:00"}
    body = client.post("/predict", json=req).json()
    assert body["arrival_time"] == "2030-11-15T21:55:00"
    assert "arrival" not in body                      # flag off: no arrival key at all
    expected = dict(SNAP["plain"]["body"], arrival_time="2030-11-15T21:55:00")
    assert body == expected


def test_arrival_block_when_flag_on(arrival_on):
    body = client.post("/predict", json=SNAP["plain"]["request"]).json()
    a = body["arrival"]
    assert a["model_version"] == "arr-ladder-v1"
    assert a["probability_late_15"] == pytest.approx(arrival_fakes.ARR_P[3], abs=1e-3)
    assert a["risk_level"] == "medium"                # 0.23: same 0.20 / 0.30 cut-offs as departure
    assert a["cancel_probability"] == pytest.approx(arrival_fakes.CANCEL_P, abs=1e-4)
    assert set(a["quantiles_minutes"]) == {"p50", "p80", "p90"}
    assert [e["minutes"] for e in a["exceedance"]][0] == -30 and len(a["exceedance"]) == 11
    assert set(a["severity"]) == {"minor_pct", "moderate_pct", "severe_pct", "expected_delay_label",
                                  "expected_delay_range"}
    # departure fields untouched
    assert body["delay_probability"] == SNAP["plain"]["body"]["delay_probability"]
    assert body["warnings"] == SNAP["plain"]["body"]["warnings"]


def test_arrival_null_and_200_when_ladder_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "ARRIVAL_FLAG", True)
    monkeypatch.setattr(main, "arrival_service",
                        arrival_service.ArrivalService(ladder_path=tmp_path / "nope.joblib"))
    r = client.post("/predict", json=SNAP["plain"]["request"])
    assert r.status_code == 200
    body = r.json()
    assert "arrival" in body and body["arrival"] is None
    assert "Arrival forecast unavailable" in body["warnings"]
    assert body["delay_probability"] == SNAP["plain"]["body"]["delay_probability"]


def test_arrival_uses_operating_carrier(arrival_on):
    """UA5521 is flown by SkyWest (OO): the ladder is built with OO's features, as the departure model is."""
    r = client.post("/predict", json=SNAP["codeshare"]["request"])
    assert r.json()["operating_airline"] == "OO"
    assert arrival_on[-1] == "OO"


def test_health_reports_arrival_model(monkeypatch):
    assert client.get("/health").json()["arrival_model_loaded"] is False
    monkeypatch.setattr(main, "arrival_service", arrival_fakes.make_service())
    assert client.get("/health").json()["arrival_model_loaded"] is True
    assert "POST /predict/itinerary" in client.get("/").json()["endpoints"]


def test_flags_default_off_and_ladder_not_loaded():
    code = "import main; print(main.ARRIVAL_FLAG, main.ITINERARY_FLAG, main.arrival_service.loaded, " \
           "main.arrival_service.load_error)"
    e = {k: v for k, v in os.environ.items() if not k.startswith("FLYRELY_FEATURE")}
    e["PYTHONPATH"] = str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e, cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False False False None"


def test_manifest_hash_mismatch_is_refused(tmp_path):
    f = tmp_path / "ladder.joblib"
    f.write_bytes(b"not the real artifact")
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"sha256": "0" * 64}))
    svc = arrival_service.ArrivalService(ladder_path=f, manifest_path=m, dependence_path=tmp_path / "d.json")
    assert not svc.loaded and "sha256" in svc.load_error


def test_dependence_loader(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"variant": "R2", "global_rho": 0.2, "by_cell": {"DEN|lt60": [0.4, 9000]}}))
    d = arrival_service.load_dependence(p)
    assert d.variant == "R2" and d.rho("DEN", 45) > 0.2


@pytest.mark.skipif(not REAL_LADDER.exists(), reason="real ladder artifact not present (gitignored, Mac only)")
def test_real_artifact_loads_and_matches_manifest():
    svc = arrival_service.ArrivalService(ladder_path=REAL_LADDER, feature_fn=main.build_features)
    assert svc.loaded, svc.load_error
    f = svc.forecast([LegSpec("SFO", "DEN", datetime(2026, 12, 3, 7, 10), "UA")])[0]
    assert 0 < f.arr.prob_late(15) < 1


def test_flags_off_keeps_unset_nested_weather_keys(monkeypatch):
    """Tomorrow.io / OpenWeatherMap weather dicts leave WeatherData.source/as_of/ceiling_ft/wx unset; master returns
    them as null, so the T6 change must not strip them (it once used response_model_exclude_unset)."""
    w = {"temperature_f": 50.0, "wind_speed_mph": 5.0, "visibility_miles": 10.0, "conditions": "clear"}

    async def fake_pair(pairs):
        return dict(w), dict(w)

    monkeypatch.setattr(main.weather_service, "get_weather_pair", fake_pair)
    body = client.post("/predict", json=SNAP["plain"]["request"]).json()
    for side in ("origin_weather", "destination_weather"):
        assert body[side] == {**w, "source": None, "as_of": None, "ceiling_ft": None, "wx": None}
    assert "arrival" not in body and "arrival_time" not in body
