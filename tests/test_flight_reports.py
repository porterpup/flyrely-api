"""POST/GET /flight-reports: auth, validation, one row per user per flight, rate limit, tester flag. Supabase mocked.
The clock is pinned (flight_reports._now), so nothing depends on today's date."""
import sqlite3
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

import admin_auth
import flight_reports
import main

client = TestClient(main.app)
REAL_ASYNC_CLIENT = httpx.AsyncClient
GOOD = {"Authorization": "Bearer good-token"}
NOW = datetime(2026, 6, 15, 18, 0, tzinfo=timezone.utc)  # 11:00 in San Francisco
DEP = "2026-06-15T08:00:00"  # SFO local, i.e. 15:00 UTC: already departed


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("FLYRELY_ADMIN_EMAILS", "boss@corp.test")
    monkeypatch.setenv("FLYRELY_FLIGHT_REPORTS_DB", str(tmp_path / "fr.sqlite"))
    monkeypatch.setattr(flight_reports, "_now", lambda: NOW)
    admin_auth.clear_cache()
    flight_reports._ready_paths.clear()
    yield
    admin_auth.clear_cache()


def as_user(monkeypatch, email="friend@gmail.test", uid="user-1", status=200):
    real = REAL_ASYNC_CLIENT
    handler = lambda req: httpx.Response(status, json={"id": uid, "email": email, "email_confirmed_at": "2026-01-01"})
    monkeypatch.setattr(admin_auth.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    admin_auth.clear_cache()


def body(**kw):
    b = {"flight_id": "UA123-SFO-2026-06-15", "airline": "UA", "flight_number": "123", "origin": "SFO",
         "dest": "JFK", "sched_dep_local": DEP, "answer": "on_time"}
    b.update(kw)
    return b


def rows():
    with sqlite3.connect(flight_reports.db_path()) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute("SELECT * FROM flight_reports ORDER BY user_id, flight_id")]


def test_no_token_is_401():
    assert client.post("/flight-reports", json=body()).status_code == 401


def test_bad_token_is_401(monkeypatch):
    as_user(monkeypatch, status=401)
    assert client.post("/flight-reports", json=body(), headers=GOOD).status_code == 401


def test_supabase_down_is_503(monkeypatch):
    as_user(monkeypatch, status=500)
    assert client.post("/flight-reports", json=body(), headers=GOOD).status_code == 503


def test_save_then_edit_is_one_row(monkeypatch):
    as_user(monkeypatch)
    r = client.post("/flight-reports", json=body(answer="on_time"), headers=GOOD)
    assert r.status_code == 200 and r.json() == {"status": "saved"}
    r = client.post("/flight-reports", json=body(answer="late_45_119"), headers=GOOD)
    assert r.status_code == 200 and r.json() == {"status": "saved"}
    got = rows()
    assert len(got) == 1
    assert got[0]["answer"] == "late_45_119" and got[0]["user_id"] == "user-1"
    assert got[0]["origin"] == "SFO" and got[0]["sched_dep_local"] == DEP


def test_withdraw_deletes_row(monkeypatch):
    as_user(monkeypatch)
    client.post("/flight-reports", json=body(), headers=GOOD)
    assert len(rows()) == 1
    r = client.post("/flight-reports", json=body(answer=None), headers=GOOD)
    assert r.status_code == 200 and r.json() == {"status": "withdrawn"}
    assert rows() == []


def test_two_users_same_flight_two_rows(monkeypatch):
    as_user(monkeypatch, uid="user-1")
    client.post("/flight-reports", json=body(), headers=GOOD)
    as_user(monkeypatch, email="other@gmail.test", uid="user-2")
    client.post("/flight-reports", json=body(answer="cancelled"), headers=GOOD)
    got = rows()
    assert [r["user_id"] for r in got] == ["user-1", "user-2"]


@pytest.mark.parametrize("bad", [
    {"answer": "very_late"}, {"airline": "U"}, {"airline": "UAAL"}, {"origin": "SF"}, {"dest": "JFKXX"},
    {"flight_id": ""}, {"flight_id": "x" * 65}, {"flight_number": "1" * 11},
    {"sched_dep_local": "2026-06-15T08:00:00+00:00"}, {"sched_dep_local": "not a date"},
])
def test_validation_422(monkeypatch, bad):
    as_user(monkeypatch)
    assert client.post("/flight-reports", json=body(**bad), headers=GOOD).status_code == 422
    assert not flight_reports.db_path().exists()


def test_future_departure_422(monkeypatch):
    as_user(monkeypatch)
    # 12:00 SFO local = 19:00 UTC, an hour after NOW
    r = client.post("/flight-reports", json=body(sched_dep_local="2026-06-15T12:00:00"), headers=GOOD)
    assert r.status_code == 422
    assert client.post("/flight-reports", json=body(sched_dep_local="2026-06-15T10:30:00"), headers=GOOD).status_code == 200


def test_more_than_30_days_old_422(monkeypatch):
    as_user(monkeypatch)
    assert client.post("/flight-reports", json=body(sched_dep_local="2026-05-10T08:00:00"), headers=GOOD).status_code == 422
    assert client.post("/flight-reports", json=body(sched_dep_local="2026-05-20T08:00:00"), headers=GOOD).status_code == 200


def test_unknown_airport_skips_future_check_but_keeps_30_day_check(monkeypatch):
    as_user(monkeypatch)
    ok = client.post("/flight-reports", json=body(origin="ZZZZ", sched_dep_local="2026-06-16T08:00:00"), headers=GOOD)
    assert ok.status_code == 200
    old = client.post("/flight-reports", json=body(origin="ZZZZ", flight_id="old", sched_dep_local="2026-05-10T08:00:00"),
                      headers=GOOD)
    assert old.status_code == 422


def test_31st_write_is_429(monkeypatch):
    as_user(monkeypatch)
    for i in range(flight_reports.RATE_LIMIT):
        assert client.post("/flight-reports", json=body(flight_id=f"f{i}"), headers=GOOD).status_code == 200
    r = client.post("/flight-reports", json=body(flight_id="f-extra"), headers=GOOD)
    assert r.status_code == 429 and "retry-after" in r.headers
    # another user is unaffected
    as_user(monkeypatch, email="other@gmail.test", uid="user-2")
    assert client.post("/flight-reports", json=body(), headers=GOOD).status_code == 200


def test_rate_limit_window_expires_and_log_is_pruned(monkeypatch):
    as_user(monkeypatch)
    for i in range(flight_reports.RATE_LIMIT):
        client.post("/flight-reports", json=body(flight_id=f"f{i}"), headers=GOOD)
    later = NOW + timedelta(hours=1, minutes=1)
    monkeypatch.setattr(flight_reports, "_now", lambda: later)
    assert client.post("/flight-reports", json=body(flight_id="again"), headers=GOOD).status_code == 200
    very_late = later + timedelta(days=8)
    monkeypatch.setattr(flight_reports, "_now", lambda: very_late)
    # departure must stay within 30 days of the new clock
    assert client.post("/flight-reports", json=body(flight_id="again", sched_dep_local="2026-06-23T08:00:00"),
                       headers=GOOD).status_code == 200
    with sqlite3.connect(flight_reports.db_path()) as c:
        assert c.execute("SELECT COUNT(*) FROM flight_report_log").fetchone()[0] == 1


def test_admin_get_403_for_normal_user(monkeypatch):
    as_user(monkeypatch)
    assert client.get("/flight-reports", headers=GOOD).status_code == 403


def test_admin_get_lists_rows(monkeypatch):
    as_user(monkeypatch)
    client.post("/flight-reports", json=body(), headers=GOOD)
    as_user(monkeypatch, email="boss@corp.test", uid="admin-1")
    r = client.get("/flight-reports", headers=GOOD)
    assert r.status_code == 200 and r.json()["count"] == 1
    assert "email" not in r.json()["items"][0]


@pytest.mark.parametrize("email,expected", [
    ("boss@corp.test", 1), ("Someone@Example.com", 1), ("friend@gmail.test", 0)])
def test_tester_flag(monkeypatch, email, expected):
    as_user(monkeypatch, email=email)
    client.post("/flight-reports", json=body(), headers=GOOD)
    assert rows()[0]["tester"] == expected


def test_no_email_stored_anywhere(monkeypatch):
    as_user(monkeypatch, email="friend@gmail.test")
    client.post("/flight-reports", json=body(), headers=GOOD)
    with sqlite3.connect(flight_reports.db_path()) as c:
        for table in ("flight_reports", "flight_report_log"):
            cols = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
            assert not {"email", "user_agent", "ip"} & cols
        dump = "\n".join(c.iterdump())
    assert "friend@gmail.test" not in dump
