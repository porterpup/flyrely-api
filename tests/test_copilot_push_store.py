"""S-A7: push subscribe/unsubscribe endpoints and the watch store (SQLite; Postgres only with TEST_DATABASE_URL)."""
import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from copilot import mount, push_store
from copilot.push_store import PushStore

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
EP = "https://web.push.apple.com/QAbc123"
FLIGHT = {"airline": "DL", "flight_number": "DL1234", "origin": "JFK", "destination": "LAX",
          "departure_time": "2026-11-15T14:30:00"}


def body(endpoint=EP, flight=None, cfid="flight-1730000000000"):
    return {"subscription": {"endpoint": endpoint, "keys": {"p256dh": "pk", "auth": "au"}},
            "flight": flight or FLIGHT, "client_flight_id": cfid}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("FLYRELY_COPILOT", "AFFILIATE_")):
            monkeypatch.delenv(k)
    monkeypatch.setenv("FLYRELY_FEATURE_DISRUPTION_COPILOT", "1")


@pytest.fixture
def store(tmp_path):
    return PushStore(sqlite_path=str(tmp_path / "p.sqlite"), database_url="")


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setenv("FLYRELY_COPILOT_PUSH", "1")
    app = FastAPI()
    mount(app)
    app.state.copilot_push_store = store
    app.state.copilot_now = lambda: NOW
    return TestClient(app)


def test_endpoints_404_unless_push_flag(store, monkeypatch):
    app = FastAPI()
    mount(app)
    app.state.copilot_push_store = store
    c = TestClient(app)
    assert c.post("/copilot/push/subscribe", json=body()).status_code == 404
    assert c.post("/copilot/push/unsubscribe", json={"endpoint": EP}).status_code == 404
    assert store.count() == 0


def test_subscribe_creates_and_is_idempotent(client, store):
    r = client.post("/copilot/push/subscribe", json=body())
    assert r.status_code == 201
    j = r.json()
    assert j["watch_id"].startswith("w_") and j["expires_at"] == "2026-11-17T02:30:00Z"
    r2 = client.post("/copilot/push/subscribe", json=body())
    assert r2.status_code == 201 and r2.json()["watch_id"] == j["watch_id"]
    assert store.count() == 1


def test_stored_row_has_no_home_airport_or_extras(client, store):
    payload = body()
    payload["home_airport"] = "SFO"   # not part of the contract: ignored, never stored
    client.post("/copilot/push/subscribe", json=payload)
    assert "home" not in " ".join(push_store.COLUMNS)
    with store._base._connect_sqlite() as conn:
        dump = " ".join(str(v) for row in conn.execute("SELECT * FROM copilot_push_watches") for v in row)
    assert "SFO" not in dump


@pytest.mark.parametrize("endpoint", [
    "http://web.push.apple.com/x", "https://evil.example.com/x", "https://169.254.169.254/latest",
    "https://localhost/x", "https://web.push.apple.com.evil.com/x", "https://user:pw@fcm.googleapis.com/x",
    "https://fcm.googleapis.com:8443/x", "https://evilfcm.googleapis.com/x", "ftp://fcm.googleapis.com/x"])
def test_ssrf_allowlist_rejects(client, store, endpoint):
    assert client.post("/copilot/push/subscribe", json=body(endpoint=endpoint)).status_code == 400
    assert store.count() == 0


@pytest.mark.parametrize("endpoint", [
    "https://web.push.apple.com/a", "https://fcm.googleapis.com/fcm/send/a",
    "https://updates.push.services.mozilla.com/wpush/v2/a", "https://wns2-par02p.notify.windows.com/w/?token=a"])
def test_allowlist_accepts_known_services(client, endpoint):
    assert client.post("/copilot/push/subscribe", json=body(endpoint=endpoint)).status_code == 201


def test_cap_of_20_watches_per_endpoint(client):
    for i in range(20):
        f = {**FLIGHT, "flight_number": f"DL{i + 1}"}
        assert client.post("/copilot/push/subscribe", json=body(flight=f)).status_code == 201
    over = {**FLIGHT, "flight_number": "DL999"}
    assert client.post("/copilot/push/subscribe", json=body(flight=over)).status_code == 429
    # an existing watch is still idempotent at the cap, and other endpoints are unaffected
    assert client.post("/copilot/push/subscribe", json=body(flight={**FLIGHT, "flight_number": "DL1"})).status_code == 201
    assert client.post("/copilot/push/subscribe", json=body(endpoint="https://fcm.googleapis.com/x", flight=over)).status_code == 201


def test_validation(client):
    bad_time = {**FLIGHT, "departure_time": "not a date"}
    assert client.post("/copilot/push/subscribe", json=body(flight=bad_time)).status_code == 422
    past = {**FLIGHT, "departure_time": "2026-11-01T10:00:00"}
    assert client.post("/copilot/push/subscribe", json=body(flight=past)).status_code == 422
    assert client.post("/copilot/push/subscribe", json=body(cfid="a b/../c")).status_code == 422
    assert client.post("/copilot/push/subscribe", json={"flight": FLIGHT}).status_code == 422


def test_unsubscribe_one_or_all(client, store):
    w1 = client.post("/copilot/push/subscribe", json=body()).json()["watch_id"]
    client.post("/copilot/push/subscribe", json=body(flight={**FLIGHT, "flight_number": "DL2"}))
    assert client.post("/copilot/push/unsubscribe", json={"endpoint": EP, "watch_id": "w_nope"}).json() == {"removed": 0}
    assert client.post("/copilot/push/unsubscribe", json={"endpoint": "https://fcm.googleapis.com/x", "watch_id": w1}).json() == {"removed": 0}
    assert client.post("/copilot/push/unsubscribe", json={"endpoint": EP, "watch_id": w1}).json() == {"removed": 1}
    assert client.post("/copilot/push/unsubscribe", json={"endpoint": EP}).json() == {"removed": 1}
    assert store.count() == 0


def test_retention_deletes_past_flights_only(store):
    dep_past, dep_future = datetime(2026, 11, 10, 9, 0), datetime(2026, 11, 16, 9, 0)
    for n, dep in (("DL1", dep_past), ("DL2", dep_future)):
        store.subscribe(endpoint=EP, p256dh="p", auth="a", airline="DL", flight_number=n, origin="JFK",
                        destination="LAX", departure=dep, now=NOW - timedelta(days=5))
    assert store.delete_expired(NOW) == 1
    assert [w.flight_number for w in store.active(NOW)] == ["DL2"]


def test_claim_is_atomic_and_release_reopens(store):
    w = store.subscribe(endpoint=EP, p256dh="p", auth="a", airline="DL", flight_number="DL1", origin="JFK",
                        destination="LAX", departure=datetime(2026, 11, 15, 14, 30), now=NOW)
    assert store.claim(w.watch_id, push_store.KIND_HEADS_UP, NOW) is True
    assert store.claim(w.watch_id, push_store.KIND_HEADS_UP, NOW) is False
    assert store.claim(w.watch_id, push_store.KIND_CANCELLED, NOW) is True   # independent kinds
    store.release(w.watch_id, push_store.KIND_HEADS_UP)
    assert store.claim(w.watch_id, push_store.KIND_HEADS_UP, NOW) is True


def test_watch_repr_hides_endpoint(store):
    w = store.subscribe(endpoint=EP, p256dh="SECRETKEY", auth="a", airline="DL", flight_number="DL1", origin="JFK",
                        destination="LAX", departure=datetime(2026, 11, 15, 14, 30), now=NOW)
    assert "apple.com" not in repr(w) and "SECRETKEY" not in repr(w)


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="Postgres round-trip needs TEST_DATABASE_URL")
def test_postgres_round_trip():
    s = PushStore(database_url=os.environ["TEST_DATABASE_URL"])
    s._exec("DELETE FROM copilot_push_watches WHERE endpoint = ?", (EP,)) if s._ready else None
    w = s.subscribe(endpoint=EP, p256dh="p", auth="a", airline="DL", flight_number="DL1", origin="JFK",
                    destination="LAX", departure=datetime(2026, 11, 15, 14, 30), now=NOW)
    assert s.claim(w.watch_id, push_store.KIND_HEADS_UP, NOW) and not s.claim(w.watch_id, push_store.KIND_HEADS_UP, NOW)
    assert s.unsubscribe(EP) >= 1
