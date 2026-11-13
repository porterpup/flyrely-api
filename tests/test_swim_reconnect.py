"""SWIM consumer reconnect safety: resource release, backoff, TLS errors, restart safety net.
Mocks only; no network, no sleeping."""
import json
import sys

import pytest

from swim import consumer, health, monitor

CERT_MSG = "Could not connect: SOLCLIENT_SUBCODE_UNTRUSTED_CERTIFICATE (certificate verify failed)"
ENV = {
    "SCDS_TFMS_URL": "tcps://broker.example.test:55443",
    "SCDS_TFMS_USER": "svc-user-xyz",
    "SCDS_TFMS_PASSWORD": "pw-Secret-123",
    "SCDS_TFMS_QUEUE": "Q/flyrely/tfms-test",
    "SCDS_TFMS_VPN": "vpn-test-name",
}


class FakeService:
    """Mock messaging service: counts live instances and records disconnect()."""
    live = 0

    def __init__(self, connect_error=None):
        self.connect_error, self.disconnected = connect_error, False
        FakeService.live += 1

    def connect(self):
        if self.connect_error:
            raise Exception(self.connect_error)

    def disconnect(self):
        if not self.disconnected:
            self.disconnected = True
            FakeService.live -= 1


@pytest.fixture
def loop_env(monkeypatch, tmp_path):
    """Run main() offline: no sleeping, no real config, recorded backoff delays."""
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    sleeps = []
    monkeypatch.setattr(consumer, "sleep_interruptible", lambda s, stats, step=5.0: sleeps.append(s))
    monkeypatch.setattr(consumer, "warm_ip_cache", lambda: None)
    monkeypatch.setattr(consumer, "open_fds", lambda: 10)
    monkeypatch.setattr(consumer, "ENV_PATH", tmp_path / "none.env")
    monkeypatch.setattr(consumer, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(consumer.signal, "signal", lambda *a: None)
    monkeypatch.setattr(consumer, "SOURCE", "tfms")
    FakeService.live = 0
    return sleeps


def run_main(tmp_path):
    return consumer.main(["--source", "tfms", "--out", str(tmp_path / "out"),
                          "--status-dir", str(tmp_path / "status")])


def status(tmp_path):
    return json.loads((tmp_path / "status" / "tfms.json").read_text())


def test_tls_error_recorded_and_backoff_grows(loop_env, monkeypatch, tmp_path):
    states = []
    real = monitor.Stats.set_state
    monkeypatch.setattr(monitor.Stats, "set_state", lambda self, st, err=None: (states.append(st), real(self, st, err)))
    monkeypatch.setattr(consumer, "build_service", lambda: FakeService(CERT_MSG))
    monkeypatch.setattr(consumer, "MAX_CONSECUTIVE_FAILURES", 6)
    assert run_main(tmp_path) == consumer.EXIT_RESTART
    assert loop_env == [5, 10, 20, 40, 80]  # the 6th failure exits instead of sleeping
    assert "tls_error" in states
    st = status(tmp_path)
    assert st["tls_reason"] == "SOLCLIENT_SUBCODE_UNTRUSTED_CERTIFICATE"
    assert st["consecutive_failures"] == 6
    assert FakeService.live == 0


def test_backoff_caps_at_ten_minutes():
    b = consumer.Backoff()
    delays = [b.next() for _ in range(12)]
    assert delays[:4] == [5, 10, 20, 40] and max(delays) == 600 and delays[-1] == 600


def test_fifty_failed_attempts_do_not_leak_services(loop_env, monkeypatch, tmp_path):
    created = []

    def build():
        created.append(FakeService("network unreachable"))
        return created[-1]

    monkeypatch.setattr(consumer, "build_service", build)
    monkeypatch.setattr(consumer, "MAX_CONSECUTIVE_FAILURES", 51)
    assert run_main(tmp_path) == consumer.EXIT_RESTART
    assert len(created) == 51 and all(s.disconnected for s in created)
    assert FakeService.live == 0
    assert max(loop_env) == 600


def test_service_released_if_failure_after_connect(monkeypatch):
    svc = FakeService()
    monkeypatch.setattr(consumer, "build_service", lambda: svc)
    monkeypatch.setitem(sys.modules, "solace.messaging.resources.queue", None)  # import fails after connect
    with pytest.raises(ImportError):
        consumer.consume(None, None, None, None)
    assert svc.disconnected


def test_backoff_resets_after_success(loop_env, monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_consume(writer, flt, stats, deadline):
        calls["n"] += 1
        if calls["n"] == 3:  # third attempt connects, then the session drops
            stats.on_connected()
        raise Exception("network unreachable")

    monkeypatch.setattr(consumer, "consume", fake_consume)
    monkeypatch.setattr(consumer, "MAX_CONSECUTIVE_FAILURES", 5)
    assert run_main(tmp_path) == consumer.EXIT_RESTART
    assert loop_env == [5, 10, 5, 10, 20, 40]  # the success on attempt 3 restarts the sequence at 5


def test_exit_after_n_consecutive_failures(loop_env, monkeypatch, tmp_path):
    def boom(*a, **k):
        raise Exception("boom")

    monkeypatch.setattr(consumer, "consume", boom)
    assert consumer.MAX_CONSECUTIVE_FAILURES == 30
    assert run_main(tmp_path) == consumer.EXIT_RESTART
    assert len(loop_env) == 29
    assert status(tmp_path)["consecutive_failures"] == 30


def test_exit_when_fds_exceed_threshold(loop_env, monkeypatch, tmp_path):
    monkeypatch.setattr(consumer, "open_fds", lambda: 9999)
    monkeypatch.setenv("SWIM_MAX_OPEN_FDS", "100")
    monkeypatch.setattr(consumer, "consume", lambda *a, **k: pytest.fail("must not connect"))
    assert run_main(tmp_path) == consumer.EXIT_RESTART


def test_tls_and_auth_classification():
    assert consumer.is_tls_error(CERT_MSG) and consumer.tls_reason(CERT_MSG).endswith("UNTRUSTED_CERTIFICATE")
    assert not consumer.is_tls_error("Login Failure") and consumer.is_auth_error("Login Failure")
    assert consumer.tls_reason("SSL handshake failed") == "TLS_ERROR"


def test_health_silence_alert_still_fires_with_tls_reason(tmp_path):
    st = monitor.Stats("tfms", tmp_path, clock=lambda: 1000.0)
    st.tls_reason = "SOLCLIENT_SUBCODE_UNTRUSTED_CERTIFICATE"
    st.set_state("tls_error", "TlsError: x")
    probs = health.check(tmp_path, ["tfms"], tmp_path, now=1000.0 + 3600, max_silence_min=10, stale_status_min=1e9)
    assert "tfms:silence" in probs and "UNTRUSTED_CERTIFICATE" in probs["tfms:silence"]
