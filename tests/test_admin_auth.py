"""/usage and /usage/export are admin-only: Supabase token check mocked, no network."""
import httpx
import pytest
from fastapi.testclient import TestClient

import admin_auth
import main

client = TestClient(main.app)
REAL_ASYNC_CLIENT = httpx.AsyncClient
GOOD = {"Authorization": "Bearer good-token"}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("FLYRELY_ADMIN_EMAILS", "Admin@Example.com, other@example.com")
    admin_auth.clear_cache()
    yield
    admin_auth.clear_cache()


class FakeSupabase:
    def __init__(self, status=200, user=None, raise_exc=None):
        self.calls = []
        self.status = status
        self.user = user if user is not None else {"email": "admin@example.com", "email_confirmed_at": "2026-01-01"}
        self.raise_exc = raise_exc

    def install(self, monkeypatch):
        outer = self
        real = REAL_ASYNC_CLIENT

        def handler(request):
            outer.calls.append(request)
            if outer.raise_exc:
                raise outer.raise_exc
            return httpx.Response(outer.status, json=outer.user)

        monkeypatch.setattr(admin_auth.httpx, "AsyncClient",
                            lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
        return self


def test_no_token_is_401():
    for path in ("/usage", "/usage/export"):
        r = client.get(path)
        assert r.status_code == 401
    assert client.get("/usage", headers={"Authorization": "Basic abc"}).status_code == 401
    assert client.get("/usage", headers={"Authorization": "Bearer "}).status_code == 401


def test_invalid_token_is_401(monkeypatch):
    FakeSupabase(status=401, user={"msg": "bad jwt"}).install(monkeypatch)
    assert client.get("/usage", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_non_admin_email_is_403(monkeypatch):
    FakeSupabase(user={"email": "someone@example.com", "email_confirmed_at": "2026-01-01"}).install(monkeypatch)
    r = client.get("/usage", headers=GOOD)
    assert r.status_code == 403
    assert r.json()["detail"] == "Admin only"
    assert client.get("/usage/export", headers=GOOD).status_code == 403


def test_admin_gets_200_case_insensitive_and_sends_anon_key(monkeypatch):
    fake = FakeSupabase().install(monkeypatch)
    r = client.get("/usage", headers=GOOD)
    assert r.status_code == 200 and "total_predictions" in r.json()
    call = fake.calls[0]
    assert str(call.url) == "https://example.supabase.co/auth/v1/user"
    assert call.headers["apikey"] == "anon-key" and call.headers["authorization"] == "Bearer good-token"
    # export: 404 only because the log is empty, which means auth passed
    assert client.get("/usage/export", headers=GOOD).status_code in (200, 404)


def test_token_result_cached_about_a_minute(monkeypatch):
    fake = FakeSupabase().install(monkeypatch)
    for _ in range(3):
        assert client.get("/usage", headers=GOOD).status_code == 200
    assert len(fake.calls) == 1
    real = admin_auth.time.monotonic
    monkeypatch.setattr(admin_auth.time, "monotonic", lambda: real() + 61)
    assert client.get("/usage", headers=GOOD).status_code == 200
    assert len(fake.calls) == 2


def test_unconfirmed_email_is_rejected(monkeypatch):
    FakeSupabase(user={"email": "admin@example.com"}).install(monkeypatch)
    assert client.get("/usage", headers=GOOD).status_code == 401


def test_fails_closed_without_config(monkeypatch):
    FakeSupabase().install(monkeypatch)
    monkeypatch.delenv("FLYRELY_ADMIN_EMAILS")
    assert client.get("/usage", headers=GOOD).status_code == 403
    monkeypatch.setenv("FLYRELY_ADMIN_EMAILS", "admin@example.com")
    monkeypatch.delenv("SUPABASE_URL")
    assert client.get("/usage", headers=GOOD).status_code == 403
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.delenv("SUPABASE_ANON_KEY")
    assert client.get("/usage", headers=GOOD).status_code == 403


def test_supabase_down_is_denied(monkeypatch):
    FakeSupabase(raise_exc=httpx.ConnectError("down")).install(monkeypatch)
    assert client.get("/usage", headers=GOOD).status_code == 503
    FakeSupabase(status=500).install(monkeypatch)
    assert client.get("/usage", headers=GOOD).status_code == 503


def test_cors_allows_authorization_header():
    r = client.options("/usage", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization"})
    assert r.status_code == 200
    assert "authorization" in r.headers["access-control-allow-headers"].lower()
