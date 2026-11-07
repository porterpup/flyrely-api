"""copilot/airlines.py and copilot/data/airlines.json. Pure file reads: no network."""

import json
import re
from pathlib import Path

import pytest

from copilot import airlines
from copilot.schemas import AirlineInfo

DATA = json.loads(Path(airlines.AIRLINES_PATH).read_text(encoding="utf-8"))["airlines"]
CARRIERS = ["AA", "DL", "UA", "WN", "AS", "B6", "NK", "F9", "G4", "HA", "SY"]  # plan §3.4 (MX, XP not in BTS data)


def test_every_plan_carrier_present():
    assert sorted(DATA) == sorted(CARRIERS)


def test_verified_entries_are_complete_and_https():
    for code, e in DATA.items():
        if e["manage_booking_url"]:
            assert e["manage_booking_url"].startswith("https://"), code
        if e["phone"]:
            assert re.fullmatch(r"\+1-\d{3}-\d{3}-\d{4}", e["phone"]), code
        if e["verified_on"] is None:
            continue
        assert e["manage_booking_url"] and e["phone"], code
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", e["verified_on"]), code
        assert e["sources"], code


def test_verification_status_valid_and_consistent():
    for code, e in DATA.items():
        assert e["verification"] in {"verified", "official_source_unconfirmed", "unknown"}, code
        assert (e["verification"] == "verified") == bool(e["verified_on"]), code
        if e["verification"] == "unknown":
            assert e["phone"] is None, code
        else:
            assert e["phone"] and e["manage_booking_url"] and e["sources"], code
        assert isinstance(e["other_channels"], dict), code


def test_unconfirmed_and_unknown_entries_have_a_note():
    for code, e in DATA.items():
        if e["verification"] != "verified":
            assert e.get("note") or e.get("user_note"), code


def test_nk_has_status_url_and_no_phone():
    e = airlines.get_entry("nk")
    assert e["phone"] is None and e["status_url"].startswith("https://")
    assert "call center" in e["user_note"] and "note" not in e


def test_lookup_returns_verified_carrier():
    info = airlines.lookup("DL")
    assert isinstance(info, AirlineInfo)
    assert info.code == "DL"
    assert info.name == "Delta Air Lines"
    assert info.manage_booking_url == "https://www.delta.com/mytrips/"
    assert info.phone == "+1-800-221-1212"
    assert info.phone_display == "800-221-1212"
    assert info.verified_on == "2026-11-06"


def test_lookup_is_case_and_whitespace_insensitive():
    assert airlines.lookup(" wn ").code == "WN"


@pytest.mark.parametrize("code", [None, "", "XX", "OO", "YX", "MQ"])
def test_unknown_or_regional_codes_return_none(code):
    assert airlines.lookup(code) is None


@pytest.mark.parametrize("code", [c for c in CARRIERS if DATA[c]["verification"] == "unknown"])
def test_unknown_carriers_return_none(code):
    assert airlines.lookup(code) is None


@pytest.mark.parametrize("code", [c for c in CARRIERS if DATA[c]["verification"] != "unknown"])
def test_shown_carriers_have_phone(code):
    info = airlines.lookup(code)
    assert info.phone == DATA[code]["phone"] and info.verification == DATA[code]["verification"]


def test_every_carrier_but_nk_shows_a_phone():
    assert [c for c in CARRIERS if not airlines.lookup(c)] == ["NK"]


def test_unconfirmed_carrier_marked_and_channels_returned():
    info = airlines.lookup("B6")
    assert info.verification == "official_source_unconfirmed"
    assert info.other_channels["x"] == "@jetblue"
    assert airlines.lookup("DL").verification == "verified"


def test_lookup_reads_given_path(tmp_path):
    p = tmp_path / "airlines.json"
    p.write_text(json.dumps({"airlines": {"ZZ": {
        "name": "Test Air", "manage_booking_url": "https://example.com/manage", "phone": "+1-555-555-0100",
        "phone_display": "555-555-0100", "verified_on": "2026-11-06"}}}), encoding="utf-8")
    airlines._load.cache_clear()
    try:
        assert airlines.lookup("zz", p).name == "Test Air"
    finally:
        airlines._load.cache_clear()
