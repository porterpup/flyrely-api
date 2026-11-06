"""S-A8: the push watcher, with a fake sender, scorer, clock and status source. No network."""
import importlib.util
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI

from copilot import config, mount, push, push_store, watcher
from copilot.push_store import PushStore
from copilot.schemas import StatusResponse
from copilot.status import SourceNotConfigured

ROOT = Path(__file__).resolve().parent.parent
EP = "https://web.push.apple.com/QAbc123"
# JFK is UTC-5 in November: 14:30 local = 19:30 UTC
DEP = datetime(2026, 11, 15, 14, 30)
T_MINUS_20H = datetime(2026, 11, 14, 23, 30, tzinfo=timezone.utc)
T_MINUS_30H = datetime(2026, 11, 14, 13, 30, tzinfo=timezone.utc)


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class Sender:
    def __init__(self, raises=None):
        self.calls, self.raises = [], raises

    def __call__(self, sub, payload):
        self.calls.append((sub, payload))
        if self.raises:
            raise self.raises


class Status:
    def __init__(self, value="unknown", raises=None):
        self.source, self.value, self.raises = "swim", value, raises

    def status(self, flight):
        if self.raises:
            raise self.raises
        return StatusResponse(status=self.value, source="swim", as_of=None, detail=None)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FLYRELY_COPILOT"):
            monkeypatch.delenv(k)


@pytest.fixture
def store(tmp_path):
    s = PushStore(sqlite_path=str(tmp_path / "p.sqlite"), database_url="")
    return s


def add(store, endpoint=EP, number="DL1234", dep=DEP, **kw):
    return store.subscribe(endpoint=endpoint, p256dh="pk", auth="au", airline="DL", flight_number=number,
                           origin="JFK", destination="LAX", departure=dep, client_flight_id="flight-1",
                           now=T_MINUS_30H - timedelta(days=1), **kw)


def make(store, sender=None, score=lambda *a: (0.34, "high"), clock=None, **kw):
    sender = sender or Sender()
    return Watcher(store, sender, score, clock, **kw), sender


def Watcher(store, sender, score, clock, **kw):
    return watcher.Watcher(store, score, sender, clock=clock or Clock(T_MINUS_20H), sleep=lambda s: None, **kw)


def test_heads_up_sent_once_for_high_risk(store):
    add(store)
    w, sender = make(store)
    assert w.run_once()["sent"] == 1
    assert w.run_once()["sent"] == 0 and w.run_once()["sent"] == 0
    assert len(sender.calls) == 1
    sub, payload = sender.calls[0]
    assert sub == {"endpoint": EP, "keys": {"p256dh": "pk", "auth": "au"}}
    assert payload["kind"] == "at_risk_heads_up" and payload["v"] == 1
    assert payload["url"] == "/flights/flight-1" and "34%" in payload["body"]
    assert "home" not in json.dumps(payload).lower() and "http" not in payload["body"]
    assert store.active(T_MINUS_20H)[0].risk_level == "high"


def test_exactly_once_across_two_watcher_instances(store):
    add(store)
    a, sender_a = make(store)
    b, sender_b = make(store)
    a.run_once(); b.run_once()
    assert len(sender_a.calls) + len(sender_b.calls) == 1


def test_not_sent_outside_window_or_when_not_high(store):
    add(store)
    w, sender = make(store, clock=Clock(T_MINUS_30H))
    assert w.run_once()["sent"] == 0                       # 30 h out
    w, sender2 = make(store, score=lambda *a: (0.10, "low"), clock=Clock(T_MINUS_20H))
    assert w.run_once()["sent"] == 0                       # in window but low risk
    w, sender3 = make(store, clock=Clock(datetime(2026, 11, 15, 20, 0, tzinfo=timezone.utc)))
    assert w.run_once()["sent"] == 0                       # already departed
    assert not (sender.calls or sender2.calls or sender3.calls)
    # rises to high later: still sent, once
    w, sender4 = make(store, clock=Clock(T_MINUS_20H))
    assert w.run_once()["sent"] == 1 and len(sender4.calls) == 1


def test_scorer_failure_is_isolated_and_retried(store):
    add(store, number="DL1"); add(store, number="DL2")
    calls = []

    def score(o, d, t, a):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("model down")
        return 0.5, "high"
    w, sender = make(store, score=score)
    s = w.run_once()
    assert s["errors"] == 1 and s["sent"] == 1
    assert w.run_once()["sent"] == 1 and len(sender.calls) == 2


def test_410_deletes_all_watches_for_endpoint(store):
    add(store, number="DL1"); add(store, number="DL2"); add(store, endpoint="https://fcm.googleapis.com/x")
    w, sender = make(store, sender=Sender(raises=push.PushGone("410")))
    s = w.run_once()
    assert s["gone"] >= 1 and s["sent"] == 0
    left = {x.endpoint for x in store.active(T_MINUS_20H)}
    assert EP not in left


def test_transient_failure_releases_claim_then_retries(store):
    add(store)
    failing = Sender(raises=push.PushError("503"))
    w, _ = make(store, sender=failing)
    assert w.run_once()["errors"] == 1
    assert store.active(T_MINUS_20H)[0].heads_up_sent_at is None
    w.sender = ok = Sender()
    assert w.run_once()["sent"] == 1 and len(ok.calls) == 1


def test_expired_watches_are_deleted(store):
    add(store, dep=datetime(2026, 11, 1, 9, 0))
    w, sender = make(store)
    assert w.run_once()["expired"] == 1 and store.count() == 0


def test_rate_limit_cap_and_delay(store):
    for i in range(5):
        add(store, number=f"DL{i + 1}")
    sleeps = []
    w = watcher.Watcher(store, lambda *a: (0.5, "high"), Sender(), clock=Clock(T_MINUS_20H), sleep=sleeps.append,
                        send_delay_s=0.25, max_sends=2)
    assert w.run_once()["sent"] == 2 and sleeps == [0.25, 0.25]
    assert w.run_once()["sent"] == 2 and w.run_once()["sent"] == 1   # the rest wait for later cycles


def test_cancelled_push_once_with_status_source(store):
    add(store)
    w, sender = make(store, score=lambda *a: (0.1, "low"), status_source=Status("cancelled"))
    assert w.run_once()["sent"] == 1 and w.run_once()["sent"] == 0
    assert sender.calls[0][1]["kind"] == "cancelled"


def test_status_stub_and_none_source_send_nothing(store):
    add(store)
    stub = Status(raises=SourceNotConfigured("R3"))
    w, sender = make(store, score=lambda *a: (0.1, "low"), status_source=stub)
    assert w.run_once()["sent"] == 0 and w.status_source is None
    w2, sender2 = make(store, score=lambda *a: (0.1, "low"),
                       status_source=type("N", (), {"source": "none"})())
    assert w2.run_once()["sent"] == 0 and not sender.calls and not sender2.calls


def test_logs_never_contain_endpoint(store, caplog):
    caplog.set_level(logging.DEBUG)
    add(store)
    make(store, sender=Sender(raises=push.PushError("push service error 503")))[0].run_once()
    make(store)[0].run_once()
    assert "apple.com" not in caplog.text and "QAbc123" not in caplog.text and "pk" not in caplog.text.split()


def test_alert_after_repeated_failures_once(store):
    add(store)
    alerts = []
    w, _ = make(store, sender=Sender(raises=push.PushError("x")), alert=lambda s, b: alerts.append(s))
    for _ in range(6):
        w.cycle_with_alerts()
    assert len(alerts) == 1
    w.sender = Sender()
    w.cycle_with_alerts()
    assert w._failures == 0 and not w._alerted


def test_alert_when_store_is_down():
    class Broken:
        def delete_expired(self, now): raise RuntimeError("db down")
    alerts = []
    w = watcher.Watcher(Broken(), None, Sender(), sleep=lambda s: None, alert=lambda s, b: alerts.append(s))
    for _ in range(3):
        assert w.cycle_with_alerts() is None
    assert len(alerts) == 1


# ---------------------------------------------------------------- the real sender, with pywebpush mocked
def test_webpush_sender_maps_errors(monkeypatch):
    import pywebpush

    class Resp:
        def __init__(self, c): self.status_code = c

    def fake(code):
        def _w(**kw):
            raise pywebpush.WebPushException("boom https://web.push.apple.com/QAbc123", response=Resp(code))
        return _w
    send = push.webpush_sender("priv", "mailto:t@example.com")
    for code, exc in ((410, push.PushGone), (404, push.PushGone), (503, push.PushError)):
        monkeypatch.setattr(pywebpush, "webpush", fake(code))
        with pytest.raises(exc) as e:
            send({"endpoint": EP, "keys": {}}, {"v": 1})
        assert "apple.com" not in str(e.value)
    seen = {}
    monkeypatch.setattr(pywebpush, "webpush", lambda **kw: seen.update(kw))
    send({"endpoint": EP, "keys": {}}, {"v": 1})
    assert seen["vapid_claims"] == {"sub": "mailto:t@example.com"} and json.loads(seen["data"]) == {"v": 1}


def test_webpush_sender_without_keys_raises_pusherror():
    with pytest.raises(push.PushError):
        push.webpush_sender()({"endpoint": EP}, {})


def test_generated_keys_are_throwaway_and_well_formed():
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "gen_vapid_keys.py")
    gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
    pub, priv = gen.generate()
    import base64
    assert len(base64.urlsafe_b64decode(pub + "==")) == 65 and len(base64.urlsafe_b64decode(priv + "==")) == 32
    assert gen.generate() != (pub, priv)
    # a real signature with the generated key, no network: py_vapid accepts the raw-scalar form
    from py_vapid import Vapid
    claims = {"sub": "mailto:t@example.com", "aud": "https://web.push.apple.com", "exp": 4102444800}
    assert Vapid.from_string(priv).sign(claims)["Authorization"].startswith("vapid ")


# ---------------------------------------------------------------- wiring: off by default, never under tests
def test_watcher_flag_helpers(monkeypatch):
    assert not config.watcher_enabled()
    monkeypatch.setenv("FLYRELY_FEATURE_DISRUPTION_COPILOT", "1")
    monkeypatch.setenv("FLYRELY_COPILOT_WATCHER", "1")
    assert not config.watcher_enabled()           # push flag still off
    monkeypatch.setenv("FLYRELY_COPILOT_PUSH", "1")
    assert config.watcher_enabled()
    monkeypatch.setenv("FLYRELY_COPILOT_WATCHER", "0")
    assert not config.watcher_enabled()           # the plan's default-on is deliberately not used: explicit "1" only


def test_mount_never_starts_watcher_under_pytest(monkeypatch):
    for k in ("FLYRELY_FEATURE_DISRUPTION_COPILOT", "FLYRELY_COPILOT_PUSH", "FLYRELY_COPILOT_WATCHER"):
        monkeypatch.setenv(k, "1")
    app = FastAPI()
    before = len(app.router.on_startup)
    mount(app)
    assert len(app.router.on_startup) == before


def test_attach_forced_runs_and_cancels_cleanly(monkeypatch, store):
    from fastapi.testclient import TestClient
    for k in ("FLYRELY_FEATURE_DISRUPTION_COPILOT", "FLYRELY_COPILOT_PUSH", "FLYRELY_COPILOT_WATCHER"):
        monkeypatch.setenv(k, "1")
    add(store)
    w, sender = make(store)
    app = FastAPI()
    assert watcher.attach(app, None, watcher=w, force=True)
    with TestClient(app):          # runs startup, then shutdown (cancels the task)
        import time
        for _ in range(50):
            if sender.calls:
                break
            time.sleep(0.05)
    assert len(sender.calls) == 1 and app.state.copilot_watcher_task.done()


def test_attach_requires_vapid_in_production_path(monkeypatch):
    for k in ("FLYRELY_FEATURE_DISRUPTION_COPILOT", "FLYRELY_COPILOT_PUSH", "FLYRELY_COPILOT_WATCHER"):
        monkeypatch.setenv(k, "1")
    assert watcher.attach(FastAPI(), None, force=True) is False


def test_cli_is_a_noop_when_flag_off():
    env = {k: v for k, v in os.environ.items() if not k.startswith("FLYRELY_")}
    env["FLYRELY_ENV_FILE"] = os.devnull
    out = subprocess.run([sys.executable, "-m", "copilot.watcher", "--once"], capture_output=True, text=True,
                         env={**env, "PYTHONPATH": str(ROOT)}, cwd=ROOT, timeout=60)
    assert out.returncode == 0 and "off" in out.stderr


def test_flag_off_app_loads_no_push_or_watcher_code():
    code = ("import sys, main\n"
            "print(any(m.startswith('copilot') or m == 'pywebpush' for m in sys.modules))")
    env = {k: v for k, v in os.environ.items() if not k.startswith("FLYRELY_")}
    env.update({"PYTHONPATH": str(ROOT), "FLYRELY_COPILOT_PUSH": "1", "FLYRELY_COPILOT_WATCHER": "1"})
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"
