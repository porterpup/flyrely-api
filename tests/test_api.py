"""Regression tests for the canonical /predict contract and its failure modes."""

from datetime import date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

CANONICAL_REQUEST = {
    "flight_id": "flight-1730000000000",
    "origin": "JFK",
    "destination": "LAX",
    "departure_time": "2026-11-15T18:30:00",
    "airline": "AA",
}

# Every field the frontend's PredictResponse type declares (src/lib/api.ts)
CANONICAL_FIELDS = {
    "flight_id", "risk_level", "delay_probability", "confidence", "delay_severity",
    "origin", "destination", "departure_time", "airline", "weather_available",
    "origin_weather", "destination_weather", "risk_factors", "recommendations",
    "warnings", "model_version",
}


@pytest.fixture(autouse=True)
def fresh_usage_log():
    main.usage_logger.log_path.unlink(missing_ok=True)
    yield
    main.usage_logger.log_path.unlink(missing_ok=True)


def test_models_load_with_matching_feature_space():
    assert main.model_service.model is not None, main.model_service.load_error
    assert main.model_service.severity_model is not None
    assert list(main.model_service.model.feature_names_in_) == main.model_service.feature_names
    built = main.build_features("JFK", "LAX", datetime(2026, 11, 15, 18, 30), "AA")
    assert set(built) == set(main.model_service.feature_names)


def test_predict_canonical_contract():
    r = client.post("/predict", json=CANONICAL_REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == CANONICAL_FIELDS
    assert body["flight_id"] == CANONICAL_REQUEST["flight_id"]
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert body["risk_level"] == main.model_service._probability_to_risk(body["delay_probability"])
    assert isinstance(body["risk_factors"], list) and isinstance(body["recommendations"], list)
    assert body["recommendations"], "recommendations must never be empty"
    assert body["weather_available"] is False  # no weather key in tests
    assert body["model_version"] == main.MODEL_VERSION

    sev = body["delay_severity"]
    assert sev is not None
    assert sum([sev["minor_pct"], sev["moderate_pct"], sev["severe_pct"]]) == pytest.approx(1.0, abs=0.01)
    assert sev["expected_delay_label"] in {"minor", "moderate", "severe"}


def test_predict_is_a_real_model_output_not_a_constant():
    probs = set()
    for hour, origin, dest in [(6, "SAN", "SLC"), (19, "EWR", "ORD"), (13, "ATL", "DFW")]:
        req = {**CANONICAL_REQUEST, "origin": origin, "destination": dest,
               "departure_time": f"2026-12-22T{hour:02d}:00:00"}
        probs.add(client.post("/predict", json=req).json()["delay_probability"])
    assert len(probs) == 3
    assert 0.5 not in probs


def test_predict_without_flight_id_generates_one():
    req = {k: v for k, v in CANONICAL_REQUEST.items() if k != "flight_id"}
    r = client.post("/predict", json=req)
    assert r.status_code == 200, r.text
    assert r.json()["flight_id"].startswith("srv-")


def test_predict_accepts_utc_suffix():
    r = client.post("/predict", json={**CANONICAL_REQUEST, "departure_time": "2026-11-15T18:30:00Z"})
    assert r.status_code == 200, r.text
    assert r.json()["departure_time"] == "2026-11-15T18:30:00"


def test_unknown_airport_is_flagged():
    r = client.post("/predict", json={**CANONICAL_REQUEST, "destination": "XYZ"})
    assert r.status_code == 200
    assert any("XYZ" in w for w in r.json()["warnings"])


def test_model_missing_returns_503_not_a_fake_number(monkeypatch):
    monkeypatch.setattr(main.model_service, "model", None)
    r = client.post("/predict", json=CANONICAL_REQUEST)
    assert r.status_code == 503
    assert r.json()["detail"]["reason"] == "model_unavailable"
    assert client.get("/health").json()["status"] == "degraded"
    assert not main.usage_logger.log_path.exists()


def test_prediction_exception_returns_503(monkeypatch):
    class Broken:
        def predict_proba(self, X):
            raise RuntimeError("boom")

    monkeypatch.setattr(main.model_service, "model", Broken())
    r = client.post("/predict", json=CANONICAL_REQUEST)
    assert r.status_code == 503
    assert r.json()["detail"]["reason"] == "prediction_failed"


def test_no_hash_or_heuristic_fallback_in_source():
    src = Path(main.__file__).read_text()
    assert "hashlib" not in src
    assert "_heuristic_predict" not in src


def test_health_shape():
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["severity_model_loaded"] is True
    assert body["weather_api_configured"] is False
    assert "timestamp" in body and "model_version" in body


def test_cors_allows_dev_origin_and_rejects_others():
    ok = client.options("/predict", headers={
        "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"})
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"

    bad = client.options("/predict", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_usage_shape_is_stable_empty_and_populated():
    keys = {"period_days", "total_predictions", "total_weather_calls", "estimated_cost_usd",
            "avg_predictions_per_day", "daily", "top_routes", "by_risk_level", "note"}
    empty = client.get("/usage").json()
    assert set(empty) == keys and empty["total_predictions"] == 0

    client.post("/predict", json=CANONICAL_REQUEST)
    full = client.get("/usage").json()
    assert set(full) == keys
    assert full["total_predictions"] == 1
    assert full["top_routes"][0]["route"] == "JFK→LAX"


def test_holiday_calendar_matches_training_dates():
    assert set(main.us_major_holidays(2025)) == {
        date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 5, 26),
        date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27), date(2025, 12, 25)}
    assert set(main.us_major_holidays(2026)) == {
        date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 5, 25),
        date(2026, 7, 4), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25)}
    # 2027 flights must not measure distance to a 2026 holiday
    assert main.get_days_to_holiday(datetime(2027, 7, 3, 12)) == 1
