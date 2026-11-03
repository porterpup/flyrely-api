"""Prediction log: one row per successful /predict, never affecting the response."""

import json
import os
import sqlite3
from datetime import datetime

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


def _record(**over):
    kw = dict(flight_id="a", origin="JFK", dest="LAX", sched_dep_local=datetime(2030, 1, 1),
              marketing_airline=None, marketing_number=None, operating_airline=None,
              operating_number=None, delay_probability=0.1, risk_level="low",
              model_version="v", features={}, weather_available=False)
    kw.update(over)
    return prediction_log.build_record(**kw)


def test_build_record_failure_leaves_response_unchanged(store, monkeypatch):
    good = client.post("/predict", json=REQ)
    before = len(rows(store))

    def boom(**kw):
        raise RuntimeError("bad record")

    monkeypatch.setattr(prediction_log, "build_record", boom)
    bad = client.post("/predict", json=REQ)
    assert bad.status_code == 200
    assert bad.json() == good.json()
    assert len(rows(store)) == before


def test_dst_boundaries():
    # Day before US fall-back (EDT, UTC-4) and after (EST, UTC-5)
    assert prediction_log.to_utc(datetime(2030, 11, 2, 12, 0), "JFK").hour == 16
    assert prediction_log.to_utc(datetime(2030, 11, 3, 12, 0), "JFK").hour == 17
    # Ambiguous 01:30 on fall-back day resolves to the first (EDT) occurrence
    assert prediction_log.to_utc(datetime(2030, 11, 3, 1, 30), "JFK").hour == 5
    # Phoenix has no DST
    assert prediction_log.to_utc(datetime(2030, 7, 1, 12, 0), "PHX").hour == 19


def test_sqlite_pragmas(store):
    store._ensure_schema()
    conn = store._connect_sqlite()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()


def test_pg_connect_uses_timeouts(monkeypatch):
    import psycopg
    calls = []

    def fake_connect(url, **kw):
        calls.append((url, kw))
        raise psycopg.OperationalError("down")

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    s = prediction_log.PredictionStore(database_url="postgresql://x/y")
    s.safe_write({})
    assert calls[0][1]["connect_timeout"] == 5
    assert "statement_timeout=5000" in calls[0][1]["options"]


def test_circuit_breaker_pauses_then_resumes(tmp_path, monkeypatch):
    s = prediction_log.PredictionStore(sqlite_path=str(tmp_path / "b.sqlite"), database_url="")
    attempts = []

    def boom(rec):
        attempts.append(1)
        raise RuntimeError("down")

    monkeypatch.setattr(s, "write", boom)
    for _ in range(prediction_log.BREAKER_THRESHOLD):
        s.safe_write({})
    assert len(attempts) == prediction_log.BREAKER_THRESHOLD
    s.safe_write({})  # paused: not attempted
    assert len(attempts) == prediction_log.BREAKER_THRESHOLD

    s._paused_until = 0.0  # cooldown elapsed
    s.safe_write({})
    assert len(attempts) == prediction_log.BREAKER_THRESHOLD + 1


def test_breaker_resets_on_success(tmp_path):
    s = prediction_log.PredictionStore(sqlite_path=str(tmp_path / "c.sqlite"), database_url="")
    s._failures = 2
    s.safe_write(_record())
    assert s._failures == 0


@pytest.mark.skipif(not os.getenv("FLYRELY_TEST_DATABASE_URL"), reason="FLYRELY_TEST_DATABASE_URL not set")
def test_postgres_roundtrip():
    import psycopg
    url = os.environ["FLYRELY_TEST_DATABASE_URL"]
    s = prediction_log.PredictionStore(database_url=url)
    rec = _record(flight_id="pgtest-1", features={"a": 1.5})
    s.write(rec)
    with psycopg.connect(url) as conn:
        row = conn.execute("SELECT client, features_json FROM predictions WHERE prediction_id = %s",
                           (rec["prediction_id"],)).fetchone()
        conn.execute("DELETE FROM predictions WHERE prediction_id = %s", (rec["prediction_id"],))
    assert row == ("app", {"a": 1.5})


def test_newly_covered_airport_gets_utc(store):
    # FWA is in models/airports_ref.csv but was not in the old 30-airport table
    r = client.post("/predict", json={**REQ, "origin": "FWA", "destination": "ATL", "flight_id": "fwa-1"})
    assert r.status_code == 200
    (row,) = rows(store)
    assert row["sched_dep_utc"] == "2030-11-15T23:30:00+00:00"  # America/Indiana/Indianapolis, EST
    assert row["minutes_to_departure"] is not None
