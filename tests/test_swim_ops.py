"""Heartbeat, health check, retention, auth parking, trust store and service installer. Offline;
nothing here touches data/ or launchctl/systemctl (the installer is only run with --dry-run/--out-dir)."""
import json
import plistlib
import sys
from datetime import date
from pathlib import Path

import pytest

from swim import archive, consumer, health, monitor, retention

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "deploy"))
import install_services as inst  # noqa: E402

NOW = 1_800_000_000.0


# ---------------------------------------------------------------- monitor
def test_stats_rates_reconnects_and_atomic_status_file(tmp_path):
    clock = [NOW]
    st = monitor.Stats("tfms", tmp_path, "gzip", clock=lambda: clock[0])
    for minute in range(5):
        for _ in range(10):
            st.message(True, "keep:flow")
        for _ in range(20):
            st.message(False, "drop:track_thinned")
        clock[0] += 60
    st.reconnect()
    st.reconnect()
    st.write(force=True)
    data = json.loads(monitor.status_path(tmp_path, "tfms").read_text())
    assert data["messages_received"] == 150 and data["messages_kept"] == 50
    assert data["msgs_per_min_5m"] == 30.0 and data["kept_per_min_5m"] == 10.0
    assert data["reconnect_count"] == 2 and len(data["reconnect_epochs_24h"]) == 2
    assert data["last_message_epoch"] and data["decisions"]["keep:flow"] == 50
    assert not list(tmp_path.glob("*.tmp"))
    text = monitor.status_path(tmp_path, "tfms").read_text()
    assert "password" not in text.lower()


def test_status_write_is_throttled_and_survives_unwritable_dir(tmp_path):
    clock = [NOW]
    st = monitor.Stats("tfms", tmp_path / "s", clock=lambda: clock[0])
    st.write()
    p = monitor.status_path(tmp_path / "s", "tfms")
    m1 = p.stat().st_mtime_ns
    st.message(True)
    st.write()  # within 15 s: no rewrite
    assert p.stat().st_mtime_ns == m1
    bad = monitor.Stats("tfms", tmp_path / "file")
    (tmp_path / "file").write_text("x")  # a file where a directory is needed
    bad.write(force=True)  # must not raise


def test_park_marker_roundtrip(tmp_path):
    monitor.park(tmp_path, "tfms", "authentication failure", now=NOW)
    assert monitor.park_path(tmp_path, "tfms").exists()
    assert monitor.unpark(tmp_path, "tfms") and not monitor.unpark(tmp_path, "tfms")


# ----------------------------------------------------------------- health
def write_status(d, src, **kw):
    base = {"source": src, "state": "connected", "updated_epoch": NOW - 10, "started_epoch": NOW - 7200,
            "last_message_epoch": NOW - 5, "reconnect_count": 0, "reconnect_epochs_24h": []}
    base.update(kw)
    monitor.write_json_atomic(monitor.status_path(d, src), base)


def chk(tmp_path, **kw):
    return health.check(tmp_path, ["tfms"], tmp_path, NOW, min_free_gb=0, **kw)


def test_health_ok(tmp_path):
    write_status(tmp_path, "tfms")
    assert chk(tmp_path) == {}


def test_health_zero_messages_for_x_minutes(tmp_path):
    write_status(tmp_path, "tfms", last_message_epoch=NOW - 15 * 60)
    assert "tfms:silence" in chk(tmp_path, max_silence_min=10)
    assert chk(tmp_path, max_silence_min=20) == {}


def test_health_new_process_gets_grace_before_first_message(tmp_path):
    write_status(tmp_path, "tfms", last_message_epoch=None, started_epoch=NOW - 120)
    assert chk(tmp_path) == {}
    write_status(tmp_path, "tfms", last_message_epoch=None, started_epoch=NOW - 3600)
    assert "tfms:silence" in chk(tmp_path)


def test_health_repeated_reconnects(tmp_path):
    write_status(tmp_path, "tfms", reconnect_epochs_24h=[NOW - 60 * i for i in range(1, 7)])
    assert "tfms:reconnects" in chk(tmp_path, reconnects=5, reconnect_window_min=30)
    write_status(tmp_path, "tfms", reconnect_epochs_24h=[NOW - 3600 * i for i in range(1, 7)])
    assert chk(tmp_path, reconnects=5, reconnect_window_min=30) == {}  # spread over hours: fine


def test_health_consumer_down_missing_and_parked(tmp_path):
    assert "tfms:nostatus" in chk(tmp_path)
    write_status(tmp_path, "tfms", updated_epoch=NOW - 1800)
    assert "tfms:down" in chk(tmp_path)
    monitor.park(tmp_path, "tfms", "authentication failure", now=NOW)
    p = chk(tmp_path)
    assert list(p) == ["tfms:parked"] and "--unpark tfms" in p["tfms:parked"]


def test_health_low_disk(tmp_path, monkeypatch):
    write_status(tmp_path, "tfms")
    monkeypatch.setattr(health, "free_gb", lambda _p: 3.0)
    assert "disk:low" in health.check(tmp_path, ["tfms"], tmp_path, NOW, min_free_gb=10)
    assert health.check(tmp_path, ["tfms"], tmp_path, NOW, min_free_gb=2) == {}


def test_alert_cooldown_and_rearm():
    probs = {"tfms:silence": "msg"}
    send, st = health.due(probs, {}, NOW, 60)
    assert send == ["msg"]
    send, st = health.due(probs, st, NOW + 600, 60)
    assert send == []  # cooling down
    send, st = health.due(probs, st, NOW + 3700, 60)
    assert send == ["msg"]
    send, st = health.due({}, st, NOW + 4000, 60)
    assert st == {}  # cleared problems are forgotten, so a recurrence alerts at once
    assert health.due(probs, st, NOW + 4100, 60)[0] == ["msg"]


def test_health_main_sends_through_alerts_and_noop_when_unset(tmp_path, monkeypatch, caplog):
    for k in ("FLYRELY_ALERT_DISCORD_WEBHOOK", "FLYRELY_ALERT_NTFY_TOPIC", "FLYRELY_ALERT_SMTP_HOST"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FLYRELY_ENV_FILE", str(tmp_path / "none.env"))
    monkeypatch.setattr(consumer, "ENV_PATH", tmp_path / "none.env")
    sent = []
    import alerts
    monkeypatch.setattr(alerts, "_post", lambda url, data, headers: sent.append((url, data)))
    args = ["--root", str(tmp_path), "--sources", "tfms", "--min-free-gb", "0"]
    assert health.main(args) == 1  # no status file: problem, but no channel configured -> no-op
    assert sent == []
    monkeypatch.setenv("FLYRELY_ALERT_NTFY_TOPIC", "t-test")
    monkeypatch.setenv("FLYRELY_ALERT_NTFY_SERVER", "https://ntfy.invalid")
    (tmp_path / "status" / "health_state.json").unlink()
    assert health.main(args) == 1 and len(sent) == 1 and sent[0][0] == "https://ntfy.invalid/t-test"
    assert health.main(args) == 1 and len(sent) == 1  # cooldown: not re-sent
    assert health.main(args + ["--unpark", "tfms"]) == 0


# --------------------------------------------------------------- retention
def make_day(root, src, day, n=2, ext=".jsonl.gz"):
    d = root / src / day
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{src}_{day.replace('-', '')}T0{i}_r{ext}").write_bytes(b"x" * 10)
    return d


def test_retention_keeps_n_days_and_never_the_newest(tmp_path):
    for day in ("2026-10-01", "2026-10-20", "2026-10-30", "2026-11-04"):
        make_day(tmp_path, "tfms", day)
    make_day(tmp_path, "sfdps", "2026-09-01")  # only day for this source: it IS the newest, so it stays
    (tmp_path / "status").mkdir()
    (tmp_path / "status" / "tfms.json").write_text("{}")
    today = date(2026, 11, 5)
    p = {d.name for d, _ in retention.plan(tmp_path, 14, today)}
    assert p == {"2026-10-01", "2026-10-20"}
    assert retention.run(tmp_path, 14, False, 0, today=today) == 0
    assert sorted(x.name for x in (tmp_path / "tfms").iterdir()) == ["2026-10-30", "2026-11-04"]
    assert (tmp_path / "sfdps" / "2026-09-01").exists() and (tmp_path / "status" / "tfms.json").exists()


def test_retention_stdds_keeps_fewer_days_by_default_and_is_overridable(tmp_path):
    for src in ("tfms", "stdds"):
        for day in ("2026-11-01", "2026-11-03", "2026-11-04", "2026-11-05"):
            make_day(tmp_path, src, day)
    today = date(2026, 11, 5)
    p = {(d.parent.name, d.name) for d, _ in retention.plan(tmp_path, 14, today)}
    assert p == {("stdds", "2026-11-01"), ("stdds", "2026-11-03")}
    p = {(d.parent.name, d.name) for d, _ in retention.plan(tmp_path, 14, today, {"stdds": 3})}
    assert p == {("stdds", "2026-11-01")}


def test_retention_newest_day_survives_even_if_ancient(tmp_path):
    make_day(tmp_path, "tfms", "2026-01-01")
    make_day(tmp_path, "tfms", "2026-01-02")
    retention.run(tmp_path, 1, False, 0, today=date(2026, 11, 5))
    assert [x.name for x in (tmp_path / "tfms").iterdir()] == ["2026-01-02"]


def test_retention_dry_run_deletes_nothing(tmp_path):
    make_day(tmp_path, "tfms", "2026-01-01")
    make_day(tmp_path, "tfms", "2026-11-04")
    retention.run(tmp_path, 14, True, 0, today=date(2026, 11, 5))
    assert len(list((tmp_path / "tfms" / "2026-01-01").iterdir())) == 2


def test_retention_only_deletes_archive_files_and_ignores_symlinks(tmp_path):
    old = make_day(tmp_path, "tfms", "2026-01-01")
    make_day(tmp_path, "tfms", "2026-11-04")
    (old / "notes.txt").write_text("keep me")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.jsonl.gz").write_bytes(b"x")
    (tmp_path / "tfms" / "2026-02-02").symlink_to(outside, target_is_directory=True)  # must not be followed
    retention.run(tmp_path, 14, False, 0, today=date(2026, 11, 5))
    assert (old / "notes.txt").exists() and not list(old.glob("*.jsonl.*"))
    assert (outside / "a.jsonl.gz").exists()


def test_retention_zst_and_validation(tmp_path):
    make_day(tmp_path, "tfms", "2026-01-01", ext=".jsonl.zst")
    make_day(tmp_path, "tfms", "2026-11-04")
    retention.run(tmp_path, 14, False, 0, today=date(2026, 11, 5))
    assert not (tmp_path / "tfms" / "2026-01-01").exists()
    with pytest.raises(ValueError):
        retention.run(tmp_path, 0, False, 0)


def test_retention_warns_when_disk_low(tmp_path, monkeypatch, caplog):
    make_day(tmp_path, "tfms", "2026-11-04")
    monkeypatch.setattr(retention, "free_gb", lambda _p: 1.5)
    with caplog.at_level("WARNING", logger="swim.retention"):
        assert retention.run(tmp_path, 14, True, 10, today=date(2026, 11, 5)) == 2
    assert "disk low" in caplog.text


# ------------------------------------------- auth failure: park, alert, no loop
@pytest.fixture
def swim_env(monkeypatch, tmp_path):
    for f, v in (("URL", "tcps://broker.example.test:55443"), ("USER", "u-test"), ("PASSWORD", "p-test"),
                 ("QUEUE", "q-test"), ("VPN", "v-test")):
        monkeypatch.setenv(f"SCDS_TFMS_{f}", v)
    monkeypatch.setenv("FLYRELY_ENV_FILE", str(tmp_path / "x.env"))
    monkeypatch.setattr(consumer, "ENV_PATH", tmp_path / "x.env")
    monkeypatch.setattr(consumer, "warm_ip_cache", lambda: None)
    monkeypatch.setenv("SWIM_BLOCKLIST_FILE", str(tmp_path / "nolist.txt"))
    monkeypatch.setattr(consumer.STOP, "is_set", lambda: False)
    return tmp_path


def run_main(tmp_path, *extra):
    return consumer.main(["--source", "tfms", "--out", str(tmp_path / "out"), "--status-dir", str(tmp_path / "st"),
                          "--exit-on-park", *extra])


def test_auth_failure_parks_alerts_once_and_never_retries(swim_env, monkeypatch):
    calls, alerts_sent = [], []

    def fake_consume(*a, **k):
        calls.append(1)
        raise consumer.AuthError("authentication failed")
    monkeypatch.setattr(consumer, "consume", fake_consume)
    monkeypatch.setattr(consumer, "alert_auth_failure", lambda s: alerts_sent.append(s))
    assert run_main(swim_env) == 3  # default exit code
    assert calls == [1] and alerts_sent == ["tfms"]
    assert monitor.park_path(swim_env / "st", "tfms").exists()
    assert json.loads(monitor.status_path(swim_env / "st", "tfms").read_text())["state"] == "parked"
    # the service manager restarts it (or the Mac reboots): it must refuse to connect again
    assert run_main(swim_env) == 3 and calls == [1]
    monitor.unpark(swim_env / "st", "tfms")
    assert run_main(swim_env) == 3 and calls == [1, 1]  # after unpark it tries once more (and parks again)


def test_transient_errors_count_reconnects_and_back_off(swim_env, monkeypatch):
    calls = []

    def flaky(writer, flt, stats, deadline):
        calls.append(1)
        if len(calls) < 4:
            raise RuntimeError("socket closed")
        consumer.STOP.set()
        return 0
    stop = consumer.threading.Event()
    monkeypatch.setattr(consumer, "STOP", stop)
    monkeypatch.setattr(consumer, "consume", flaky)
    monkeypatch.setattr(stop, "wait", lambda s: None)  # no real sleeping
    assert run_main(swim_env) == 0 and len(calls) == 4
    st = json.loads(monitor.status_path(swim_env / "st", "tfms").read_text())
    assert st["reconnect_count"] == 3 and st["state"] == "stopped"
    assert "p-test" not in json.dumps(st)


def test_self_check_is_offline_and_prints_no_values(swim_env, capsys):
    rc = run_main(swim_env, "--check")
    out = capsys.readouterr().out
    assert "OK" in out and "p-test" not in out and "broker.example" not in out
    assert rc in (0, 2)  # 2 only if solace-pubsubplus is not installed in this environment


# ------------------------------------------------------------- trust store
def test_trust_store_linux_uses_system_dir(monkeypatch, tmp_path):
    certs = tmp_path / "certs"
    certs.mkdir()
    (certs / "a.pem").write_text("x")
    monkeypatch.delenv("SCDS_TRUST_STORE_DIR", raising=False)
    monkeypatch.setattr(consumer, "LINUX_CERT_DIR", certs)
    monkeypatch.setattr(consumer, "_TRUST_DIR", None)
    monkeypatch.setattr(sys, "platform", "linux")
    assert consumer.trust_store_dir() == str(certs)
    monkeypatch.setenv("SCDS_TRUST_STORE_DIR", "/override")
    assert consumer.trust_store_dir() == "/override"


def test_trust_store_macos_splits_certifi_once(monkeypatch):
    monkeypatch.delenv("SCDS_TRUST_STORE_DIR", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(consumer, "_TRUST_DIR", None)
    d = consumer.trust_store_dir()
    assert d == consumer.trust_store_dir() and list(Path(d).glob("*.pem"))
    import shutil
    shutil.rmtree(d)


# ------------------------------------------------------------------- codec
def test_codec_selection(monkeypatch):
    assert archive.resolve_codec("gzip") == "gzip"
    monkeypatch.setattr(archive, "zstd_module", lambda: None)
    assert archive.resolve_codec("auto") == "gzip"  # zstandard missing (e.g. no wheel on an old Mac)
    with pytest.raises(ValueError):
        archive.resolve_codec("zstd")
    with pytest.raises(ValueError):
        archive.resolve_codec("lz4")


def test_zstd_stream_is_one_file_per_hour_and_smaller_than_gzip(tmp_path):
    pytest.importorskip("zstandard")
    from datetime import datetime, timezone
    ts = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    line = (json.dumps({"props": {"a": "b" * 200}, "body": "<x>" + "payload " * 100 + "</x>"}) + "\n").encode()
    sizes = {}
    for codec in ("gzip", "zstd"):
        w = archive.ArchiveWriter(tmp_path / codec, "tfms", "r", codec)
        for i in range(200):
            w.write_batch([(ts, line + str(i).encode() + b"\n")])
        w.close()
        files = list((tmp_path / codec).glob("*/*"))
        assert len(files) == 1
        sizes[codec] = files[0].stat().st_size
        assert len(list(archive.read_lines(files[0]))) == 400
    assert sizes["zstd"] <= sizes["gzip"]


def test_hour_rollover_closes_previous_stream(tmp_path):
    from datetime import datetime, timezone
    w = archive.ArchiveWriter(tmp_path, "tfms", "r", "gzip")
    t1 = datetime(2026, 1, 2, 3, 59, 59, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, 4, 0, 1, tzinfo=timezone.utc)
    w.write_batch([(t1, b"a\n"), (t2, b"b\n")])
    assert len(w._streams) == 1  # the 03h stream was finished
    w.close()
    import gzip
    names = sorted(p.name for p in (tmp_path / "2026-01-02").iterdir())
    assert len(names) == 2 and all(gzip.open(tmp_path / "2026-01-02" / n).read() for n in names)


# --------------------------------------------------------------- installer
def opts(*argv):
    return inst.parse(list(argv))


def test_installer_renders_macos_agent_and_daemon_plists(tmp_path):
    for mode in ("agent", "daemon"):
        o = opts("--platform", "macos", "--mode", mode, "--user", "seans", "--python", "/opt/venv/bin/python",
                 "--repo", str(tmp_path), "--sources", "tfms,sfdps", "--out-dir", str(tmp_path / mode))
        files = inst.build(o)
        assert [f[1] for f in files] == ["com.flyrely.swim.tfms", "com.flyrely.swim.sfdps",
                                         "com.flyrely.swim.health", "com.flyrely.swim.retention"]
        for _, label, path, content in files:
            pl = plistlib.loads(content.encode())  # valid plist, no leftover placeholders
            assert pl["Label"] == label and "__" not in content
            assert ("UserName" in pl) == (mode == "daemon")
        svc = plistlib.loads(files[0][3].encode())
        assert svc["KeepAlive"] is True and svc["ThrottleInterval"] >= 30
        args = svc["ProgramArguments"]
        assert "--exit-on-park" not in args  # the service parks in-process; it never exits on auth/config errors
        assert args[:3] == ["/opt/venv/bin/python", "-m", "swim.consumer"] and "tfms" in args
        assert svc["EnvironmentVariables"]["FLYRELY_ENV_FILE"] == str(tmp_path.resolve() / ".env")


def test_installer_renders_systemd_units(tmp_path):
    o = opts("--platform", "linux", "--mode", "system", "--user", "ubuntu", "--python", "/srv/venv/bin/python",
             "--repo", str(tmp_path), "--out-dir", str(tmp_path / "u"))
    files = {f[1]: f[3] for f in inst.build(o)}
    assert set(files) == {"flyrely-swim-tfms.service", "flyrely-swim-sfdps.service", "flyrely-swim-health.service",
                          "flyrely-swim-health.timer", "flyrely-swim-retention.service", "flyrely-swim-retention.timer"}
    svc = files["flyrely-swim-tfms.service"]
    assert "RestartPreventExitStatus=2 3" in svc and "Restart=on-failure" in svc and "User=ubuntu" in svc
    assert "SCDS_TRUST_STORE_DIR=/etc/ssl/certs" in svc and "--source tfms" in svc
    assert 'ExecStart="/srv/venv/bin/python" -m swim.consumer' in svc and 'Environment="FLYRELY_ENV_FILE=' in svc
    assert "__" not in "".join(files.values())
    user_mode = {f[1]: f[3] for f in inst.build(opts("--platform", "linux", "--mode", "user", "--repo", str(tmp_path)))}
    assert "User=" not in user_mode["flyrely-swim-tfms.service"] and "default.target" in user_mode["flyrely-swim-tfms.service"]


def test_installer_dry_run_changes_nothing_and_is_idempotent(tmp_path, capsys, monkeypatch):
    ran = []
    monkeypatch.setattr(inst.subprocess, "run", lambda *a, **k: ran.append(a))
    base = ["--platform", "linux", "--mode", "user", "--repo", str(tmp_path), "--out-dir", str(tmp_path / "units"),
            "--log-dir", str(tmp_path / "logs")]
    assert inst.main(base + ["--dry-run"]) == 0
    assert not (tmp_path / "units").exists() and not (tmp_path / "logs").exists() and ran == []
    assert "dry run complete" in capsys.readouterr().out
    assert inst.main(base + ["--no-load"]) == 0
    first = {p.name: p.read_text() for p in (tmp_path / "units").iterdir()}
    capsys.readouterr()
    assert inst.main(base + ["--no-load"]) == 0
    out = capsys.readouterr().out
    states = [ln.split()[0] for ln in out.splitlines() if ln.startswith(("new", "changed", "unchanged"))]
    assert states == ["unchanged"] * len(first)
    assert {p.name: p.read_text() for p in (tmp_path / "units").iterdir()} == first
    assert ran == []  # --no-load never calls systemctl


def test_installer_rejects_bad_arguments(tmp_path):
    for bad in (["--platform", "macos", "--mode", "system"], ["--sources", "foo"], ["--retention-days", "0"],
                ["--platform", "linux", "--mode", "daemon"]):
        with pytest.raises(SystemExit):
            opts(*bad)


def test_installer_loads_services_with_launchctl_commands(tmp_path, monkeypatch):
    cmds = []

    def fake_run(c, **k):
        cmds.append(c)
        return type("R", (), {"returncode": 1 if c[1] == "print" else 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(inst.subprocess, "run", fake_run)
    monkeypatch.setattr(inst, "launchd_domain", lambda o: "system")
    o = opts("--platform", "macos", "--mode", "daemon", "--user", "seans", "--repo", str(tmp_path),
             "--out-dir", str(tmp_path / "d"), "--sources", "tfms", "--log-dir", str(tmp_path / "l"))
    inst.install(o)
    flat = [" ".join(c) for c in cmds]
    assert any(c.startswith("launchctl bootstrap system ") and c.endswith("com.flyrely.swim.tfms.plist") for c in flat)
    assert "launchctl enable system/com.flyrely.swim.tfms" in flat


def test_health_status_summary(tmp_path, capsys):
    write_status(tmp_path / "status", "tfms", msgs_per_min_5m=12.5)
    assert health.main(["--root", str(tmp_path), "--sources", "tfms,sfdps", "--status"]) == 0
    out = capsys.readouterr().out
    assert "tfms: connected received/min=" in out and "sfdps: no status file" in out


# ------------------------------------- consume() end to end with a fake broker
class FakeMsg:
    def __init__(self, props, body, i):
        self.props, self.body, self.i = props, body, i

    def get_payload_as_bytes(self): return self.body.encode()
    def get_payload_as_string(self): return self.body
    def get_properties(self): return dict(self.props)
    def is_redelivered(self): return False
    def get_replication_group_message_id(self): return f"id-{self.i}"
    def get_application_message_id(self): return None
    def get_sender_timestamp(self): return 1


def test_consume_filters_archives_then_acks_everything(tmp_path, monkeypatch):
    import types
    from test_swim_filters import FLIGHT, tfms
    events = []
    msgs = [FakeMsg(FLIGHT, tfms("UAL1", "FlightTimes"), 1),          # kept
            FakeMsg(FLIGHT, tfms("N21DP", "FlightTimes"), 2),         # dropped: not an airline
            FakeMsg({"TFMDataClass": "FlowInformation"}, "<flow/>", 3)]  # kept

    class Receiver:
        def receive_message(self, timeout):
            if msgs:
                return msgs.pop(0)
            consumer.STOP.set()
            return None
        def ack(self, m): events.append(("ack", m.i, writer_state()))
        def start(self): pass
        def terminate(self, grace_period=0): pass

    class Service:
        def connect(self): pass
        def disconnect(self): pass
        def create_persistent_message_receiver_builder(self):
            return types.SimpleNamespace(build=lambda q: Receiver())

    class Queue:
        durable_non_exclusive_queue = staticmethod(lambda name: name)
    mods = {"solace": types.ModuleType("solace"), "solace.messaging": types.ModuleType("solace.messaging"),
            "solace.messaging.resources": types.ModuleType("r"),
            "solace.messaging.resources.queue": types.SimpleNamespace(Queue=Queue),
            "solace.messaging.errors": types.ModuleType("e"),
            "solace.messaging.errors.pubsubplus_client_error": types.SimpleNamespace(PubSubPlusClientError=RuntimeError)}
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.setattr(consumer, "build_service", lambda: Service())
    monkeypatch.setenv("SCDS_TFMS_QUEUE", "q")
    stop = consumer.threading.Event()
    monkeypatch.setattr(consumer, "STOP", stop)
    writer = archive.ArchiveWriter(tmp_path / "out", "tfms", "r", "gzip")

    def writer_state():  # at ack time the data must already be readable from disk (fsync before ack)
        files = list((tmp_path / "out").glob("*/*.jsonl.gz"))
        return len([r for f in files for r in archive.read_lines(f)])
    from swim import filters
    stats = monitor.Stats("tfms", tmp_path / "st", "gzip")
    n = consumer.consume(writer, filters.MessageFilter("tfms"), stats, None)
    writer.close()
    assert n == 3 and [e[:2] for e in events] == [("ack", 1), ("ack", 2), ("ack", 3)]
    assert all(e[2] == 2 for e in events)  # both kept records were on disk before the first ack
    from swim import inspect_tfms
    recs = list(inspect_tfms.iter_records(tmp_path / "out"))
    assert len(recs) == 2 and recs[0]["dedupe"]["rgmid"] == "id-1"
    assert stats.received == 3 and stats.kept == 2 and stats.reasons["drop:not_airline"] == 1
    assert json.loads(monitor.status_path(tmp_path / "st", "tfms").read_text())["state"] == "connected"


# ------------------------------------------------ in-process parking
def test_auth_park_waits_in_process_and_resumes_after_unpark(swim_env, monkeypatch):
    import threading
    stop = threading.Event()
    monkeypatch.setattr(consumer, "STOP", stop)
    monkeypatch.setattr(consumer, "PARK_POLL_S", 0.05)
    monkeypatch.setattr(consumer, "alert_auth_failure", lambda s: None)
    calls = []

    def fake_consume(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise consumer.AuthError("authentication failed")
        stop.set()  # second attempt (after unpark): stop the test
        return 0
    monkeypatch.setattr(consumer, "consume", fake_consume)
    st = swim_env / "st"
    seen = {}

    def operator():
        for _ in range(200):
            if monitor.park_path(st, "tfms").exists():
                break
            threading.Event().wait(0.02)
        threading.Event().wait(0.3)  # the process must be WAITING, not retrying
        seen["calls_while_parked"] = len(calls)
        seen["state"] = json.loads(monitor.status_path(st, "tfms").read_text())["state"]
        monitor.unpark(st, "tfms")
    t = threading.Thread(target=operator)
    t.start()
    rc = consumer.main(["--source", "tfms", "--out", str(swim_env / "out"), "--status-dir", str(st)])  # no --exit-on-park
    t.join()
    assert rc == 0 and seen == {"calls_while_parked": 1, "state": "parked"} and len(calls) == 2


def test_config_error_parks_in_process_with_status(swim_env, monkeypatch):
    import threading
    monkeypatch.delenv("SCDS_TFMS_PASSWORD")
    stop = threading.Event()
    monkeypatch.setattr(consumer, "STOP", stop)
    monkeypatch.setattr(consumer, "PARK_POLL_S", 0.05)
    monkeypatch.setattr(consumer, "consume", lambda *a, **k: pytest.fail("must not connect"))
    threading.Timer(0.3, stop.set).start()
    rc = consumer.main(["--source", "tfms", "--out", str(swim_env / "out"), "--status-dir", str(swim_env / "st")])
    assert rc == 0  # exits only because it was stopped (SIGTERM)
    st = json.loads(monitor.status_path(swim_env / "st", "tfms").read_text())
    assert st["state"] == "config_error" and "SCDS_TFMS_PASSWORD" in st["last_error"] and "p-test" not in json.dumps(st)
    assert "tfms:config" in health.check(swim_env / "st", ["tfms"], swim_env, st["updated_epoch"], min_free_gb=0)


def test_config_error_exit_code_in_manual_mode(swim_env, monkeypatch):
    monkeypatch.delenv("SCDS_TFMS_USER")
    assert run_main(swim_env) == 2


def test_log_file_disables_console_duplicate(tmp_path):
    import logging
    root = logging.getLogger()
    saved = list(root.handlers)
    try:
        consumer.setup_logging(tmp_path / "l" / "swim.log")
        kinds = [type(h).__name__ for h in root.handlers]
        assert kinds == ["RotatingFileHandler"]  # nothing goes to stderr, so launchd's stderr file stays tiny
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved:
            root.addHandler(h)


# -------------------------------------------------- write error handling
def test_write_error_abandons_file_and_continues_in_a_new_one(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from swim import inspect_tfms
    ts = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    w = archive.ArchiveWriter(tmp_path, "tfms", "r", "gzip")
    rec = lambda n: [(ts, consumer.encode_record(ts, {}, f"<a n='{n}'/>".encode()))]
    w.write_batch(rec(1))
    real = archive.durable_sync
    calls = {"n": 0}

    def flaky(fd):
        calls["n"] += 1
        raise OSError(5, "I/O error")
    monkeypatch.setattr(archive, "durable_sync", flaky)
    with pytest.raises(OSError):
        w.write_batch(rec(2))  # raised: the caller must not ack this batch
    monkeypatch.setattr(archive, "durable_sync", real)
    w.write_batch(rec(3))
    w.close()
    files = sorted(p.name for p in tmp_path.glob("*/*.jsonl.gz"))
    assert len(files) == 2 and files[1].endswith("_s1.jsonl.gz")
    bodies = [r["body"] for r in inspect_tfms.iter_records(tmp_path)]
    assert "<a n='1'/>" in bodies and "<a n='3'/>" in bodies  # later data is readable despite the torn earlier file


def test_durable_sync_uses_fullfsync_on_macos_and_falls_back(monkeypatch, tmp_path):
    import types
    monkeypatch.setattr(archive.sys, "platform", "darwin")
    seen = []
    fake = types.SimpleNamespace(F_FULLFSYNC=51, fcntl=lambda fd, op: seen.append(("fcntl", op)))
    monkeypatch.setitem(sys.modules, "fcntl", fake)
    monkeypatch.setattr(archive.os, "fsync", lambda fd: seen.append(("fsync",)))
    archive.durable_sync(3)
    assert seen == [("fcntl", 51)]
    fake.fcntl = lambda fd, op: (_ for _ in ()).throw(OSError(25, "not supported"))
    archive.durable_sync(3)
    assert seen[-1] == ("fsync",)
    # directories take the same path
    seen.clear()
    fake.fcntl = lambda fd, op: seen.append(("fcntl", op))
    archive._fsync_dir(tmp_path)
    assert seen == [("fcntl", 51)]


def test_installer_dry_run_does_not_claim_loaded(tmp_path, capsys):
    o = opts("--platform", "macos", "--mode", "agent", "--repo", str(tmp_path), "--out-dir", str(tmp_path / "x"),
             "--sources", "tfms", "--dry-run")
    inst.install(o)
    out = capsys.readouterr().out
    assert "already loaded" not in out and "launchctl enable" in out and out.index("launchctl enable") < out.index("launchctl bootstrap")


def test_installer_reload_waits_and_enables_before_bootstrap(tmp_path, monkeypatch):
    cmds, state = [], {"loaded": True}

    def fake_run(c, **k):
        cmds.append(" ".join(c[1:3]))
        if c[1] == "print":
            return type("R", (), {"returncode": 0 if state["loaded"] else 1, "stdout": "", "stderr": ""})()
        if c[1] == "bootout":
            state["loaded"] = False
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(inst.subprocess, "run", fake_run)
    monkeypatch.setattr(inst, "launchd_domain", lambda o: "gui/501")
    monkeypatch.setattr(inst.time, "sleep", lambda s: None)
    o = opts("--platform", "macos", "--mode", "agent", "--repo", str(tmp_path), "--out-dir", str(tmp_path / "x"),
             "--sources", "tfms", "--log-dir", str(tmp_path / "l"))
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "com.flyrely.swim.tfms.plist").write_text("old")  # changed file: must reload
    inst.install(o)
    seq = [c for c in cmds if c.startswith(("bootout", "enable", "bootstrap"))]
    assert seq.index("bootout gui/501/com.flyrely.swim.tfms") < seq.index("enable gui/501/com.flyrely.swim.tfms") \
        < seq.index("bootstrap gui/501")
    assert cmds.index("print gui/501/com.flyrely.swim.tfms") < cmds.index("bootout gui/501/com.flyrely.swim.tfms")


def test_installer_uninstall_reloads_systemd(tmp_path, monkeypatch):
    cmds = []
    monkeypatch.setattr(inst.subprocess, "run", lambda c, **k: (cmds.append(" ".join(c)), type("R", (), {"returncode": 0})())[1])
    o = opts("--platform", "linux", "--mode", "user", "--repo", str(tmp_path), "--out-dir", str(tmp_path / "u"), "--uninstall")
    inst.main(["--platform", "linux", "--mode", "user", "--repo", str(tmp_path), "--out-dir", str(tmp_path / "u"), "--uninstall"])
    assert cmds[-1] == "systemctl --user daemon-reload"


def test_systemd_paths_with_spaces_and_percent_are_quoted(tmp_path):
    repo = tmp_path / "my repo %h"
    repo.mkdir()
    o = opts("--platform", "linux", "--mode", "user", "--repo", str(repo), "--python", str(repo / "venv" / "bin" / "python"))
    svc = {f[1]: f[3] for f in inst.build(o)}["flyrely-swim-tfms.service"]
    assert f'ExecStart="{str(repo).replace("%", "%%")}/venv/bin/python"' in svc
    with pytest.raises(SystemExit):
        opts("--platform", "linux", "--mode", "user", "--repo", str(tmp_path), "--python", '/a"b/python')
