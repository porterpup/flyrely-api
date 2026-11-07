"""Dedup state logic for ntfy alerts (offline: urlopen is patched)."""
import json
from unittest import mock

from swim import health, ntfy

ENV = {"FLYRELY_NTFY_TOPIC": "t-test"}


def test_transitions_open_dedupe_recover():
    o, r, st = ntfy.transitions({"a": "A down"}, {})
    assert o == ["A down"] and r == [] and st == {"a": "A down"}
    o, r, st = ntfy.transitions({"a": "A down"}, st)
    assert o == [] and r == []
    o, r, st = ntfy.transitions({}, st)
    assert o == [] and r == ["A down"] and st == {}


def test_notify_sends_once_then_recovers(tmp_path):
    sf = tmp_path / "s.json"
    with mock.patch.object(ntfy, "send", return_value=True) as s:
        ntfy.notify_transitions({"a": "A down"}, sf, ENV)
        ntfy.notify_transitions({"a": "A down"}, sf, ENV)
        assert s.call_count == 1
        ntfy.notify_transitions({}, sf, ENV)
        assert s.call_count == 2 and "recovered" in s.call_args[0][1]
    assert json.loads(sf.read_text()) == {}


def test_failed_send_retries_next_run(tmp_path):
    sf = tmp_path / "s.json"
    with mock.patch.object(ntfy, "send", return_value=False) as s:
        assert ntfy.notify_transitions({"a": "x"}, sf, ENV) is False
    assert ntfy.load_state(sf) == {}
    with mock.patch.object(ntfy, "send", return_value=True) as s:
        ntfy.notify_transitions({"a": "x"}, sf, ENV)
        assert s.call_count == 1


def test_no_topic_logs_only_and_keeps_state(tmp_path):
    sf = tmp_path / "s.json"
    with mock.patch.object(ntfy, "send") as s:
        assert ntfy.notify_transitions({"a": "x"}, sf, {}) is False
        s.assert_not_called()
    assert not sf.exists()
    assert ntfy.send("t", "b", env={}) is False


def test_corrupt_state_is_empty(tmp_path):
    sf = tmp_path / "s.json"
    sf.write_text("{nope")
    assert ntfy.load_state(sf) == {}


def test_check_http_reports_failure():
    p = health.check_http({"api": "http://127.0.0.1:9/health"}, timeout=1, retries=0)
    assert "http:api" in p
