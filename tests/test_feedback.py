"""POST/GET /feedback: auth, validation, storage, rate limit, ntfy failure tolerance. Supabase mocked."""
import httpx
import pytest
from fastapi.testclient import TestClient

import admin_auth
import feedback
import main

client = TestClient(main.app)
REAL_ASYNC_CLIENT = httpx.AsyncClient
GOOD = {"Authorization": "Bearer good-token"}


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("FLYRELY_ADMIN_EMAILS", "admin@example.com")
    monkeypatch.setenv("FLYRELY_FEEDBACK_DB", str(tmp_path / "fb.sqlite"))
    monkeypatch.delenv("FLYRELY_NTFY_TOPIC", raising=False)
    admin_auth.clear_cache()
    feedback._ready_paths.clear()
    yield
    admin_auth.clear_cache()


def as_user(monkeypatch, email="friend@example.com", status=200):
    real = REAL_ASYNC_CLIENT
    handler = lambda req: httpx.Response(status, json={"email": email, "email_confirmed_at": "2026-01-01"})
    monkeypatch.setattr(admin_auth.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_no_token_is_401():
    assert client.post("/feedback", json={"message": "hi"}).status_code == 401


def test_bad_token_is_401(monkeypatch):
    as_user(monkeypatch, status=401)
    assert client.post("/feedback", json={"message": "hi"}, headers=GOOD).status_code == 401


def test_post_ok_and_stored(monkeypatch):
    as_user(monkeypatch)
    body = {"message": "  prediction looks off ", "category": "wrong prediction",
            "context": {"route": "/flight/x", "app_version": "abc",
                        "flight": {"airline": "UA", "flight_number": "123", "origin": "SFO",
                                   "destination": "JFK", "delay_probability": 0.42, "risk_level": "medium"}}}
    r = client.post("/feedback", json=body, headers=GOOD)
    assert r.status_code == 201
    as_admin = client
    # read back as admin
    admin_auth.clear_cache()
    as_user(monkeypatch, email="admin@example.com")
    items = as_admin.get("/feedback", headers=GOOD).json()["items"]
    assert len(items) == 1
    it = items[0]
    assert it["email"] == "friend@example.com" and it["message"] == "prediction looks off"
    assert it["category"] == "wrong prediction"
    assert it["context"]["flight"]["origin"] == "SFO" and it["context"]["route"] == "/flight/x"


@pytest.mark.parametrize("body", [
    {"message": ""}, {"message": "   "}, {"message": "x" * 2001},
    {"message": "ok", "category": "rant"}, {},
])
def test_validation(monkeypatch, body):
    as_user(monkeypatch)
    assert client.post("/feedback", json=body, headers=GOOD).status_code == 422


def test_max_length_accepted(monkeypatch):
    as_user(monkeypatch)
    assert client.post("/feedback", json={"message": "x" * 2000}, headers=GOOD).status_code == 201


def test_rate_limit_per_user(monkeypatch):
    as_user(monkeypatch)
    for _ in range(feedback.RATE_LIMIT):
        assert client.post("/feedback", json={"message": "m"}, headers=GOOD).status_code == 201
    assert client.post("/feedback", json={"message": "m"}, headers=GOOD).status_code == 429
    admin_auth.clear_cache()
    as_user(monkeypatch, email="other@example.com")
    assert client.post("/feedback", json={"message": "m"}, headers={"Authorization": "Bearer t2"}).status_code == 201


def test_get_requires_admin(monkeypatch):
    assert client.get("/feedback").status_code == 401
    as_user(monkeypatch)
    assert client.get("/feedback", headers=GOOD).status_code == 403


def test_ntfy_failure_ignored(monkeypatch):
    as_user(monkeypatch)
    monkeypatch.setenv("FLYRELY_NTFY_TOPIC", "t-opic")
    calls = []

    def boom(url, **kw):
        calls.append((url, kw))
        raise httpx.ConnectError("down")

    monkeypatch.setattr(feedback.httpx, "post", boom)
    r = client.post("/feedback", json={"message": "hello " * 40}, headers=GOOD)
    assert r.status_code == 201
    assert calls[0][0] == "https://ntfy.sh/t-opic"
    assert calls[0][1]["content"].decode().startswith("New feedback from friend@example.com: hello")
    assert len(calls[0][1]["content"].decode()) < 160


def test_ntfy_skipped_without_topic(monkeypatch):
    as_user(monkeypatch)
    monkeypatch.setattr(feedback.httpx, "post", lambda *a, **k: pytest.fail("should not post"))
    assert client.post("/feedback", json={"message": "m"}, headers=GOOD).status_code == 201
