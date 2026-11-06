"""S-A5: status sources (copilot/status.py) and home/away context (copilot/context.py)."""

import csv
from pathlib import Path

import pytest

from copilot import context as ctx
from copilot import status as st
from copilot.schemas import FlightRef, StatusResponse

ROOT = Path(__file__).resolve().parent.parent
AIRPORTS_REF = ROOT / "models" / "airports_ref.csv"


def _flight(origin="JFK", destination="LAX") -> FlightRef:
    return FlightRef(airline="AA", flight_number="AA117", origin=origin, destination=destination,
                     departure_time="2026-11-15T14:30:00")


# ---- status sources -------------------------------------------------------------------------------------------

def test_null_source_is_unknown():
    result = st.NullStatusSource().status(_flight())
    assert result == StatusResponse(status="unknown", source="none", as_of=None, detail=None)


def test_swim_stub_raises_and_names_roadmap_r3():
    with pytest.raises(st.SourceNotConfigured, match="R3"):
        st.SwimLiveStateSource().status(_flight())
    assert "R3" in (st.SwimLiveStateSource.__doc__ or "")


def test_factory_unset_env_gives_null_source(monkeypatch):
    monkeypatch.delenv("FLYRELY_COPILOT_STATUS_SOURCE", raising=False)
    assert isinstance(st.get_status_source(), st.NullStatusSource)


@pytest.mark.parametrize("value", ["none", "NONE", " none "])
def test_factory_none_env(monkeypatch, value):
    monkeypatch.setenv("FLYRELY_COPILOT_STATUS_SOURCE", value)
    assert isinstance(st.get_status_source(), st.NullStatusSource)


def test_factory_swim_env(monkeypatch):
    monkeypatch.setenv("FLYRELY_COPILOT_STATUS_SOURCE", "swim")
    source = st.get_status_source()
    assert isinstance(source, st.SwimLiveStateSource)
    with pytest.raises(st.SourceNotConfigured):
        source.status(_flight())


def test_factory_unknown_env_falls_back_to_none(monkeypatch):
    # config.status_source() normalises unknown values to "none", so the factory never sees them.
    monkeypatch.setenv("FLYRELY_COPILOT_STATUS_SOURCE", "bogus")
    assert isinstance(st.get_status_source(), st.NullStatusSource)


def test_factory_explicit_unknown_name_raises():
    with pytest.raises(ValueError):
        st.get_status_source("bogus")


# ---- home / away context --------------------------------------------------------------------------------------

def test_no_home_airport_is_unknown():
    result = ctx.location(None, "JFK")
    assert (result.location, result.reason, result.home_airport) == ("unknown", "no_home_airport", None)


@pytest.mark.parametrize("home", ["", "   "])
def test_blank_home_airport_is_unknown(home):
    result = ctx.location(home, "JFK")
    assert result.location == "unknown" and result.reason == "no_home_airport"


def test_origin_is_home():
    result = ctx.location("BOS", "BOS")
    assert (result.location, result.reason, result.home_airport) == ("home", "origin_is_home", "BOS")


def test_origin_not_home():
    result = ctx.location("BOS", "JFK")
    assert (result.location, result.reason, result.home_airport) == ("away", "origin_not_home", "BOS")


@pytest.mark.parametrize("home,origin", [(" bos ", "BOS"), ("Bos", "bos ")])
def test_codes_are_normalised(home, origin):
    assert ctx.location(home, origin).location == "home"


def test_origin_required():
    with pytest.raises(ValueError):
        ctx.location("BOS", "")


# Round trip: outbound leg starts at home, return leg starts away. One-way: a single leg from home or away.
@pytest.mark.parametrize("leg_origin,expected", [
    ("BOS", "home"),   # round trip outbound, or one-way from home
    ("LAX", "away"),   # round trip return, or a one-way from away
])
def test_round_trip_and_one_way_legs(leg_origin, expected):
    assert ctx.location("BOS", leg_origin).location == expected


def test_round_trip_each_leg_is_evaluated_separately():
    outbound = ctx.location("BOS", "BOS")   # BOS -> LAX
    returning = ctx.location("BOS", "LAX")  # LAX -> BOS
    assert outbound.location == "home"
    assert returning.location == "away"


def test_multi_city_each_leg_uses_its_own_origin():
    # Home BOS. Legs: BOS->LAX, LAX->SFO, SFO->SEA, SEA->BOS.
    legs = [("BOS", "home"), ("LAX", "away"), ("SFO", "away"), ("SEA", "away")]
    assert [ctx.location("BOS", origin).location for origin, _ in legs] == [expected for _, expected in legs]


def test_layover_does_not_change_the_result():
    # A connection BOS -> ORD -> LAX is one FlightRef with origin BOS. ORD is not a leg origin, so it is ignored.
    # A connection from a layover city is its own leg with origin ORD, which counts as away from BOS.
    assert ctx.location("BOS", "BOS").location == "home"
    assert ctx.location("BOS", "ORD").location == "away"


# Metro areas: several nearby airports count as home.

@pytest.mark.parametrize("home,origin", [
    ("JFK", "LGA"), ("LGA", "EWR"), ("EWR", "JFK"),
    ("ORD", "MDW"), ("MDW", "ORD"),
    ("DCA", "IAD"), ("BWI", "DCA"),
    ("SFO", "OAK"), ("SJC", "SFO"),
    ("LAX", "BUR"), ("LGB", "SNA"), ("ONT", "LAX"),
    ("MIA", "FLL"), ("FLL", "MIA"),
    ("DFW", "DAL"), ("DAL", "DFW"),
    ("IAH", "HOU"), ("HOU", "IAH"),
])
def test_metro_group_members_count_as_home(home, origin):
    result = ctx.location(home, origin)
    assert (result.location, result.reason) == ("home", "origin_is_home")


@pytest.mark.parametrize("home,origin", [
    ("JFK", "BOS"),   # NYC metro does not include Boston
    ("LAX", "SFO"),   # LA and Bay Area are separate metros
    ("DFW", "IAH"),   # Dallas and Houston are separate metros
    ("MIA", "TPA"),   # Tampa is not in the South Florida group
    ("BOS", "PVD"),   # neighbouring airports outside any group are away
])
def test_airports_outside_the_metro_are_away(home, origin):
    assert ctx.location(home, origin).location == "away"


def test_metro_groups_use_only_airports_in_reference_file():
    with AIRPORTS_REF.open(newline="") as f:
        known = {row["iata"] for row in csv.DictReader(f)}
    members = {code for codes in ctx.METRO_GROUPS.values() for code in codes}
    assert members <= known, sorted(members - known)


def test_each_airport_is_in_at_most_one_metro_group():
    seen: dict[str, str] = {}
    for group, codes in ctx.METRO_GROUPS.items():
        for code in codes:
            assert code not in seen, f"{code} in {seen[code]} and {group}"
            seen[code] = group
