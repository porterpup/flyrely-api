"""SWIM consumer reconnect safety: resource release, backoff, TLS errors, restart safety net.
Mocks only; no network, no sleeping."""
import json
import socket
import sys
import threading

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


def test_tls_error_recorded_then_restart_after_two(loop_env, monkeypatch, tmp_path):
    states = []
    real = monitor.Stats.set_state
    monkeypatch.setattr(monitor.Stats, "set_state", lambda self, st, err=None: (states.append(st), real(self, st, err)))
    monkeypatch.setattr(consumer, "build_service", lambda: FakeService(CERT_MSG))
    assert run_main(tmp_path) == consumer.EXIT_RESTART
    assert loop_env == [5]  # 2nd consecutive TLS error exits for a clean restart instead of waiting
    assert "tls_error" in states
    st = status(tmp_path)
    assert st["tls_reason"] == "SOLCLIENT_SUBCODE_UNTRUSTED_CERTIFICATE"
    assert st["consecutive_failures"] == 2
    assert FakeService.live == 0


def test_backoff_grows_on_repeated_failures(loop_env, monkeypatch, tmp_path):
    monkeypatch.setattr(consumer, "build_service", lambda: FakeService("network unreachable"))
    monkeypatch.setenv("SWIM_MAX_CONSECUTIVE_FAILURES", "6")
    assert run_main(tmp_path) == consumer.EXIT_RESTART
    assert loop_env == [5, 10, 20, 40, 80]


def test_bad_env_values_fall_back(loop_env, monkeypatch, tmp_path):
    monkeypatch.setenv("SWIM_MAX_CONSECUTIVE_FAILURES", "abc")
    monkeypatch.setenv("SWIM_MAX_OPEN_FDS", "abc")
    monkeypatch.setattr(consumer, "build_service", lambda: FakeService("network unreachable"))
    assert run_main(tmp_path) == consumer.EXIT_RESTART
    assert len(loop_env) == 29  # default 30


def test_hostname_mismatch_is_a_config_error_not_a_retry_loop(loop_env, monkeypatch, tmp_path):
    built = []
    real_msg = ("sub_code 'SOLCLIENT_SUBCODE_UNTRUSTED_CERTIFICATE' ... Certificate verification failed, "
                "rc='hostname mismatch'")
    monkeypatch.setattr(consumer, "build_service", lambda: built.append(1) or FakeService(real_msg))
    rc = consumer.main(["--source", "tfms", "--out", str(tmp_path / "out"),
                        "--status-dir", str(tmp_path / "status"), "--exit-on-park"])
    assert rc == 2 and built == [1] and loop_env == []
    assert status(tmp_path)["state"] == "config_error"


RESET_MSG = ("sub_code 'SOLCLIENT_SUBCODE_COMMUNICATION_ERROR' ... \"SSL 'SSL-client' cannot read, sslErr = 5: "
             "'Connection reset by peer (54)\"")


def test_connection_reset_is_transient_not_tls(loop_env, monkeypatch, tmp_path):
    assert not consumer.is_tls_error(RESET_MSG)
    states = []
    real = monitor.Stats.set_state
    monkeypatch.setattr(monitor.Stats, "set_state", lambda self, st, err=None: (states.append(st), real(self, st, err)))
    monkeypatch.setattr(consumer, "build_service", lambda: FakeService(RESET_MSG))
    monkeypatch.setenv("SWIM_MAX_CONSECUTIVE_FAILURES", "5")
    assert run_main(tmp_path) == consumer.EXIT_RESTART  # via the 5-failure path, not the 2-TLS-error path
    assert loop_env == [5, 10, 20, 40]  # normal growing backoff
    assert "tls_error" not in states and status(tmp_path)["tls_reason"] is None


def test_tls_fast_exit_recorded_then_suppressed_within_30_min(loop_env, monkeypatch, tmp_path):
    monkeypatch.setattr(consumer, "build_service", lambda: FakeService(CERT_MSG))
    assert run_main(tmp_path) == consumer.EXIT_RESTART and loop_env == [5]
    first = status(tmp_path)["last_tls_exit_epoch"]
    assert first is not None
    loop_env.clear()
    # "launchd restart" soon after: the same TLS error must not fast-exit again
    monkeypatch.setenv("SWIM_MAX_CONSECUTIVE_FAILURES", "5")
    assert run_main(tmp_path) == consumer.EXIT_RESTART
    assert loop_env == [5, 10, 20, 40]  # normal backoff and the 5-failure path
    assert status(tmp_path)["last_tls_exit_epoch"] == first


def test_tls_fast_exit_allowed_again_after_cooldown(loop_env, monkeypatch, tmp_path):
    import time as _t
    monitor.write_json_atomic(tmp_path / "status" / "tfms.json",
                              {"last_tls_exit_epoch": _t.time() - consumer.TLS_EXIT_COOLDOWN_S - 5})
    monkeypatch.setattr(consumer, "build_service", lambda: FakeService(CERT_MSG))
    assert run_main(tmp_path) == consumer.EXIT_RESTART and loop_env == [5]


def test_silence_detected_across_restart(tmp_path):
    t_last = 1_000_000.0
    monitor.write_json_atomic(tmp_path / "tfms.json", {
        "last_message_epoch": t_last, "started_epoch": t_last - 100, "updated_epoch": t_last + 60,
        "state": "reconnecting", "restart_epochs_24h": []})
    now = t_last + 3600
    st = monitor.Stats("tfms", tmp_path, clock=lambda: now)  # a new process: started_epoch == now
    st.carry_forward()
    probs = health.check(tmp_path, ["tfms"], tmp_path, now=now + 30, max_silence_min=10)
    assert "tfms:silence" in probs
    assert json.loads((tmp_path / "tfms.json").read_text())["restart_epochs_24h"] == [round(now)]


def test_health_alerts_on_tls_state_and_restart_rate(tmp_path):
    now = 5_000_000.0
    monitor.write_json_atomic(tmp_path / "tfms.json", {
        "state": "restarting", "tls_reason": "X", "updated_epoch": now, "started_epoch": now,
        "last_message_epoch": now, "restart_epochs_24h": [now - 60, now - 600, now - 1200]})
    probs = health.check(tmp_path, ["tfms"], tmp_path, now=now)
    assert "tfms:tls" in probs and "tfms:restarts" in probs


def test_trust_store_refreshed_once_per_process_start(monkeypatch, tmp_path):
    monkeypatch.delenv("SCDS_TRUST_STORE_DIR", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(consumer, "TRUST_ROOT", tmp_path / "trust")
    monkeypatch.setattr(consumer, "SOURCE", "tfms")
    monkeypatch.setattr(consumer, "_TRUST_REFRESHED", set())
    d = consumer.trust_store_dir()
    (tmp_path / "trust" / "tfms" / "stale.pem").write_text("x")
    assert consumer.trust_store_dir() == d and (tmp_path / "trust" / "tfms" / "stale.pem").exists()  # no rebuild
    monkeypatch.setattr(consumer, "_TRUST_REFRESHED", set())  # "new process"
    consumer.trust_store_dir()
    assert not (tmp_path / "trust" / "tfms" / "stale.pem").exists()  # repaired
    assert list((tmp_path / "trust" / "tfms").glob("*.pem"))
    assert [p.name for p in (tmp_path / "trust").iterdir()] == ["tfms"]  # no temp siblings left


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


class AlreadyDisconnectedService:
    """Mimics solace 1.11: after a failed connect the state is DISCONNECTED, so disconnect() returns early and
    only session_cleanup() frees the context."""

    def __init__(self, cleanup_raises=False):
        self.cleaned = 0
        outer = self

        class Session:
            def session_cleanup(self):
                outer.cleaned += 1
                if cleanup_raises:
                    raise RuntimeError("boom")

        self._session = Session()

    def connect(self):
        raise Exception("Connection refused")

    def disconnect(self):
        return None  # "already disconnected": nothing freed


@pytest.mark.parametrize("cleanup_raises", [False, True])
def test_session_cleanup_called_even_when_disconnect_returns_early(monkeypatch, cleanup_raises):
    svc = AlreadyDisconnectedService(cleanup_raises)
    monkeypatch.setattr(consumer, "build_service", lambda: svc)
    with pytest.raises(Exception, match="Connection refused"):  # a cleanup failure must not mask the real error
        consumer.consume(None, None, None, None)
    assert svc.cleaned == 1


def test_release_tolerates_services_without_a_session():
    consumer._release(object(), None)  # no disconnect, no _session: must not raise


def test_real_library_refused_connection_does_not_leak(monkeypatch, tmp_path):
    pytest.importorskip("solace.messaging.messaging_service")
    if consumer.open_fds() < 0:
        pytest.skip("no fd listing on this platform")
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]  # closed again on exit: nothing listens, so connect is refused
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("SCDS_TFMS_URL", f"tcps://127.0.0.1:{port}")
    monkeypatch.setenv("SCDS_TRUST_STORE_DIR", str(tmp_path))
    (tmp_path / "ca.pem").write_text("x")
    monkeypatch.setattr(consumer, "SOURCE", "tfms")

    def attempt():
        with pytest.raises(Exception):
            consumer.consume(None, None, None, None)

    attempt()  # warm-up: one-time library fds/threads
    fds0, th0 = consumer.open_fds(), threading.active_count()
    for _ in range(5):
        attempt()
    assert consumer.open_fds() <= fds0 + 1
    assert threading.active_count() <= th0 + 1
