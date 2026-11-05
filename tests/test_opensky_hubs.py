"""Pure logic in backtest/opensky_hubs.py: box sizing, block splitting, departure/arrival event detection."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
import opensky_hubs as oh  # noqa: E402

LAT, LON, ELEV = 40.0, -75.0, 10.0


def _rows(spec):
    # spec: (t, lat_offset_deg, ground, alt_agl, vr)
    return pd.DataFrame([("XXX", t, t, "abc123", "DAL100", t, LON, LAT + dl, ELEV + a if a is not None else None,
                          ELEV + a if a is not None else None, g, vr) for t, dl, g, a, vr in spec],
                        columns=["hub", "rt", "t", "icao24", "cs", "te", "lon", "lat", "balt", "galt", "ground", "vr"])


def _detect(spec):
    oh.HUB_COORDS["XXX"] = (LAT, LON, ELEV)
    return oh.detect_events(_rows(spec), "XXX", [])


def test_box_is_one_credit():
    b = oh.box(40.0, -75.0)
    area = (b["lamax"] - b["lamin"]) * (b["lomax"] - b["lomin"])
    assert area <= 25   # /states/all costs 1 credit up to 25 square degrees


def test_departure_bracketed_by_ground_then_climb():
    e = _detect([(1000, 0.0, True, None, 0), (1120, 0.02, False, 300, 8)])
    assert list(e.kind) == ["dep"] and bool(e.bracketed.iloc[0]) and e.bracket_s.iloc[0] == 120
    assert 1000 <= e.t.iloc[0] <= 1120


def test_departure_first_seen_climbing_unbracketed():
    e = _detect([(2000, 0.03, False, 200, 6)])
    assert list(e.kind) == ["dep"] and not bool(e.bracketed.iloc[0])


def test_arrival_bracketed_and_unbracketed():
    a = _detect([(3000, 0.05, False, 400, -4), (3100, 0.0, True, None, 0)])
    assert list(a.kind) == ["arr"] and bool(a.bracketed.iloc[0])
    u = _detect([(4000, 0.05, False, 300, -4)])
    assert list(u.kind) == ["arr"] and not bool(u.bracketed.iloc[0])


def test_overflight_is_not_an_event():
    assert _detect([(5000, 0.05, False, 5000, 0), (5120, 0.07, False, 5000, 0)]).empty


def test_blocks_split_on_gap():
    assert oh.blocks_from([0, 120, 240, 40000, 40120]) == [(0, 240), (40000, 40120)]


def test_cost_table_and_caps():
    t = oh.cost_table()
    assert t["every_2min"]["credits_per_hub_24h"] == 720 and t["every_2min"]["hubs_fitting_in_4000_24h"] == 5
    assert oh.STATES_CAP - oh.STATES_BEFORE_PHASE == 1500


def test_classify_hub_applies_preregistered_rule():
    ok = {"n": 50, "median": 130.0}
    assert oh.classify_hub(0.95, 0.9, ok, ok) == "reliable"
    assert oh.classify_hub(0.95, 0.9, None, ok) == "partial"   # no bracketed departures: timing not verifiable
    assert oh.classify_hub(0.95, 0.9, {"n": 50, "median": 200.0}, ok) == "partial"   # bracket median above 3 minutes
    assert oh.classify_hub(0.8, 0.9, ok, ok) == "partial"
    assert oh.classify_hub(0.95, 0.65, ok, ok) == "not usable"


def test_airline_callsign_pattern_is_icao_plus_one_to_four_digits():
    assert oh.AL_RE.match("DAL1234") and oh.AL_RE.match("SWA7")
    assert not oh.AL_RE.match("DAL12345") and not oh.AL_RE.match("N123AB") and not oh.AL_RE.match("DAL12A")
