"""Consumer safety tests: redaction, tcps-only, compression, crash tolerance. Synthetic values; no network."""
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from swim import consumer
from swim import inspect_tfms as it

SECRETS = {
    "SCDS_TFMS_URL": "tcps://Broker-One.example.test:55443,tcps://broker-two.example.test:55443",
    "SCDS_TFMS_USER": "svc-user-xyz",
    "SCDS_TFMS_PASSWORD": "pw-Secret-123",
    "SCDS_TFMS_QUEUE": "Q/flyrely/tfms-test",
    "SCDS_TFMS_VPN": "vpn-test-name",
}
TS = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@pytest.fixture
def secrets(monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setitem(consumer._IP_CACHE, "Broker-One.example.test", ("203.0.113.7",))


def test_redact_all_fields_case_insensitive_multihost_ip(secrets):
    text = ("fail user=SVC-USER-XYZ pw=pw-Secret-123 vpn=VPN-TEST-NAME q=q/flyrely/tfms-test "
            "host=BROKER-TWO.example.test:55443 host1=broker-one.EXAMPLE.test ip=203.0.113.7")
    out = consumer.redact(text).lower()
    for v in ("svc-user-xyz", "pw-secret-123", "vpn-test-name", "tfms-test", "broker-two",
              "broker-one", "203.0.113.7"):
        assert v not in out


@pytest.mark.parametrize("field", ["USER", "VPN", "QUEUE", "PASSWORD"])
def test_redact_each_field(secrets, field):
    v = SECRETS[f"SCDS_TFMS_{field}"]
    assert v not in consumer.redact(f"x {v} y") and v.upper() not in consumer.redact(f"x {v.upper()} y")


def test_logging_filter_redacts_library_lines(secrets, capsys):
    logger = logging.getLogger("solace.test")
    h = logging.StreamHandler()
    h.addFilter(consumer.RedactingFilter())
    logger.addHandler(h)
    logger.propagate = False
    try:
        logger.warning("connecting to %s as %s", "broker-two.example.test", "svc-user-xyz")
    finally:
        logger.removeHandler(h)
    err = capsys.readouterr().err
    assert "broker-two" not in err and "svc-user-xyz" not in err and "***" in err


def test_setup_logging_quiets_solace_and_filters():
    consumer.setup_logging()
    assert logging.getLogger("solace").level == logging.WARNING
    assert all(any(isinstance(f, consumer.RedactingFilter) for f in h.filters)
               for h in logging.getLogger().handlers)


def test_require_tcps():
    assert consumer.require_tcps("tcps://a.example:1,TCPS://b.example:1")
    for bad in ("tcp://a.example:55555", "smf://a.example", "a.example:55443",
                "tcps://a.example,tcp://b.example", ""):
        with pytest.raises(consumer.ConfigError):
            consumer.require_tcps(bad)


def test_compression_level_refuses_zero(monkeypatch):
    monkeypatch.setenv("SCDS_COMPRESSION_LEVEL", "0")
    with pytest.raises(consumer.ConfigError):
        consumer.compression_level()
    monkeypatch.setenv("SCDS_COMPRESSION_LEVEL", "5")
    assert consumer.compression_level() == 5


def test_auth_detection_is_specific():
    assert consumer.is_auth_error("SOLCLIENT_SUBCODE_LOGIN_FAILURE")
    assert consumer.is_auth_error("Authentication failed")
    assert not consumer.is_auth_error("connection timed out after 4031 ms")
    assert not consumer.is_auth_error("error code 40300")


def test_truncated_final_chunk_is_tolerated(tmp_path):
    def rec(i):
        return (TS, consumer.encode_record(TS, {}, f"<a n='{i}'/>".encode()))
    consumer.write_batch(tmp_path, [rec(1), rec(2)])
    consumer.write_batch(tmp_path, [rec(3)])
    f = next(tmp_path.glob("*/*.jsonl.gz"))
    f.write_bytes(f.read_bytes()[:-10])  # simulate a crash mid-chunk
    assert [r["body"] for r in it.iter_records(tmp_path)] == ["<a n='1'/>", "<a n='2'/>"]


def test_dedupe_fields_and_per_run_file(tmp_path):
    rec = json.loads(consumer.encode_record(TS, {}, b"hello", {"rgmid": "r1", "app_id": None, "sender_ts": 5}))
    assert rec["dedupe"]["rgmid"] == "r1" and rec["dedupe"]["sender_ts"] == 5
    assert len(rec["dedupe"]["sha256"]) == 64
    assert consumer.RUN_ID in consumer.hour_path(tmp_path, TS).name


def test_inspector_property_allowlist():
    assert "solace_routing_dest_name" not in it.ALLOWED_PROPS
    assert "MessageType" in it.ALLOWED_PROPS


def test_trust_store_built_once(monkeypatch):
    monkeypatch.delenv("SCDS_TRUST_STORE_DIR", raising=False)
    monkeypatch.setattr(consumer, "_TRUST_DIR", None)
    a = consumer.trust_store_dir()
    assert a == consumer.trust_store_dir()
    assert list(Path(a).glob("*.pem"))
    shutil.rmtree(a)
