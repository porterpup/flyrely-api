"""Prediction log: one row per successful /predict, never affecting the response."""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import main
import prediction_log

client = TestClient(main.app)

REQ = {
    "flight_id": "flight-1",
    "origin": "JFK",
    "destination": "LAX",
    "departure_time": "2030-11-15T18:30:00",
    "airline": "AA",
    "flight_number": "100",
}


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = prediction_log.PredictionStore(sqlite_path=str(tmp_path / "p.sqlite"), database_url="")
    monkeypatch.setattr(main, "prediction_store", s)
    return s


def rows(store):
    conn = sqlite3.connect(store.sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM predictions")]
    finally:
        conn.close()


def test_one_row_with_all_fields(store):
    r = client.post("/predict", json=REQ)
    assert r.status_code == 200
    body = r.json()
    (row,) = rows(store)
    assert set(row) == set(prediction_log.COLUMNS)
    assert row["flight_id"] == "flight-1"
    assert row["client"] == "app"
    assert (row["origin"], row["dest"]) == ("JFK", "LAX")
    assert row["sched_dep_local"] == "2030-11-15T18:30:00"
    assert row["sched_dep_utc"] == "2030-11-15T23:30:00+00:00"  # EST = UTC-5
    assert row["marketing_airline"] == "AA" and row["marketing_number"] == body["flight_number"]
    assert row["operating_airline"] == body["operating_airline"]
    assert row["operating_number"] == body["operating_flight_number"]
    assert row["horizon"] == "booking"
    assert row["minutes_to_departure"] > 0
    assert row["delay_probability"] == body["delay_probability"]
    assert row["risk_level"] == body["risk_level"]
    assert row["model_version"] == body["model_version"] == main.MODEL_VERSION
    assert row["rates_as_of"] is None and row["live_inputs_json"] is None
    assert row["weather_available"] == 0
    assert row["ts_utc"] and row["prediction_id"]
    assert set(json.loads(row["features_json"])) == set(main.model_service.feature_names)


def test_unknown_airport_tz_leaves_utc_null(store):
    r = client.post("/predict", json={**REQ, "origin": "XYZ", "flight_id": "x-1"})
    assert r.status_code == 200
    (row,) = rows(store)
    assert row["sched_dep_utc"] is None and row["minutes_to_departure"] is None


def test_h2h_client_tag(store):
    client.post("/predict", json={**REQ, "flight_id": "h2h-abc"})
    client.post("/predict", json={**REQ, "flight_id": "flight-2"})
    tags = {r["flight_id"]: r["client"] for r in rows(store)}
    assert tags == {"h2h-abc": "h2h", "flight-2": "app"}


def test_broken_store_does_not_change_response(store, monkeypatch):
    good = client.post("/predict", json=REQ)

    def boom(rec):
        raise RuntimeError("db down")

    monkeypatch.setattr(store, "write", boom)
    bad = client.post("/predict", json=REQ)
    assert bad.status_code == 200
    assert bad.json() == good.json()


def test_503_writes_nothing(store, monkeypatch):
    monkeypatch.setattr(main.model_service, "model", None)
    r = client.post("/predict", json=REQ)
    assert r.status_code == 503
    assert not store.sqlite_path.exists() or rows(store) == []
