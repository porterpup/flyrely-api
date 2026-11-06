"""S-A6: POST /copilot/options and /status, with a fake scorer, a fixture schedule and a frozen clock."""
import gzip
import logging
import os
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from copilot import mount
from copilot.schedule import BtsTypicalSchedule
from copilot.status import NullStatusSource, SwimLiveStateSource

HEADER = ("marketing_airline,marketing_flight_number,origin,dest,dep_time,arr_time,operating_airline,"
          "operating_flight_number,dow_mask,days_seen,schedule_as_of\n")
ROWS = [
    "AA,117,JFK,LAX,16:00,19:25,AA,117,127,8,2026-08",
    "DL,1234,JFK,LAX,14:30,18:00,DL,1234,127,8,2026-08",       # the user's own flight
    "DL,50,JFK,LAX,07:15,10:40,DL,50,127,6,2026-08",
    "UA,900,JFK,LAX,12:00,15:30,OO,5521,127,4,2026-08",
    "B6,23,JFK,LAX,21:00,00:30,B6,23,127,5,2026-08",           # arrives next day
    "AA,1,BOS,JFK,09:00,10:15,AA,1,127,5,2026-08",
]
FLIGHT_DAY = "2026-11-15"   # a Sunday


def fake_score(origin, dest, dep, airline):
    return {"AA": (0.10, "low"), "UA": (0.40, "medium"), "B6": (0.10, "low")}.get(airline, (0.70, "high"))


def noon_utc(day="2026-11-15", hour=12):
    d = datetime.fromisoformat(f"{day}T{hour:02d}:00:00").replace(tzinfo=timezone.utc)
    return lambda: d


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("FLYRELY_COPILOT", "AFFILIATE_")):
            monkeypatch.delenv(k)
    monkeypatch.setenv("FLYRELY_FEATURE_DISRUPTION_COPILOT", "1")


@pytest.fixture
def make_client(tmp_path):
    p = tmp_path / "route_schedule.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write(HEADER + "\n".join(ROWS) + "\n")

    def make(score=fake_score, now=None, status_source=None):
        app = FastAPI()
        mount(app, score=score)
        app.state.copilot_schedule = BtsTypicalSchedule(p)
        # now = 09:00 New York time on the flight day (14:00 UTC), before every fixture departure but 07:15
        app.state.copilot_now = now or noon_utc(hour=14)
        if status_source:
            app.state.copilot_status_source = status_source
        return TestClient(app)
    return make


def body(**over):
    b = {"flight": {"airline": "DL", "flight_number": "DL1234", "origin": "JFK", "destination": "LAX",
                    "departure_time": f"{FLIGHT_DAY}T14:30:00"},
         "state": "cancelled", "home_airport": "BOS"}
    b.update(over)
    return b


def test_away_cancelled_full_response(make_client):
    r = make_client().post("/copilot/options", json=body())
    assert r.status_code == 200
    j = r.json()
    assert j["context"] == {"location": "away", "reason": "origin_not_home", "home_airport": "BOS"}
    assert j["airline"]["code"] == "DL"
    assert j["refund_rights"] == {"applies": True, "jurisdiction": "US_DOT", "content_key": "us_dot_part260_v1"}
    alt = j["alternatives"]
    assert alt["source"] == "bts_typical_schedule" and alt["schedule_as_of"] == "2026-08" and alt["date"] == FLIGHT_DAY
    nums = [i["flight_number"] for i in alt["items"]]
    assert "DL1234" not in nums                      # own flight excluded
    assert "DL50" not in nums                        # 07:15 local has already departed (clock is 09:00 local)
    assert nums == ["AA117", "B6" + "23", "UA900"]   # prob asc, ties by departure time
    ua = alt["items"][2]
    assert ua["operating_airline"] == "OO" and ua["risk_level"] == "medium" and ua["delay_probability"] == 0.4
    assert alt["items"][1]["arr_local"] == "2026-11-16T00:30:00"
    assert [h["partner"] for h in alt["search_handoffs"]] == ["google_flights", "kayak", "skyscanner", "expedia"]
    assert alt["items"][0]["handoffs"] == alt["search_handoffs"]
    assert j["hotel"]["airport"] == "JFK" and j["hotel"]["check_in"] == FLIGHT_DAY
    assert j["hotel"]["check_out"] == "2026-11-16"
    assert j["ride"]["purpose"] == "to_hotel" and [x["partner"] for x in j["ride"]["links"]] == ["uber", "lyft"]
    assert j["ride"]["links"][0]["accepts_dropoff_address"] is True
    assert j["disclosure"] == {"required": False}
    assert j["warnings"] == []


def test_home_has_no_hotel_and_home_ride(make_client):
    j = make_client().post("/copilot/options", json=body(home_airport="LGA")).json()   # same NYC metro
    assert j["context"]["location"] == "home"
    assert j["hotel"] is None
    assert j["ride"]["purpose"] == "home" and j["ride"]["links"]


def test_unknown_location_no_ride_no_hotel(make_client):
    j = make_client().post("/copilot/options", json=body(home_airport=None)).json()
    assert j["context"] == {"location": "unknown", "reason": "no_home_airport", "home_airport": None}
    assert j["hotel"] is None and j["ride"] == {"purpose": None, "links": []}


def test_at_risk_returns_same_list_and_no_hotel(make_client):
    c = make_client()
    a = c.post("/copilot/options", json=body(state="at_risk")).json()
    b = c.post("/copilot/options", json=body()).json()
    assert a["alternatives"]["items"] == b["alternatives"]["items"]
    assert a["hotel"] is None


def test_rebooked_today_vs_later(make_client):
    c = make_client()
    today = c.post("/copilot/options", json=body(state="rebooked", rebooked="today")).json()
    assert today["hotel"] is None and today["ride"] == {"purpose": None, "links": []}
    later = c.post("/copilot/options", json=body(state="rebooked", rebooked="later")).json()
    assert later["hotel"] is not None and later["ride"]["purpose"] == "to_hotel"


def test_hotel_check_in_today_when_flight_is_tomorrow_night(make_client):
    c = make_client(now=noon_utc("2026-11-15", 4))   # 23:00 local on the 14th... 04:00Z = 23:00 EST on the 14th
    j = c.post("/copilot/options", json=body()).json()
    assert j["hotel"]["check_in"] == "2026-11-14" and j["hotel"]["check_out"] == FLIGHT_DAY


def test_no_alternatives(make_client):
    b = body()
    b["flight"].update(origin="LAX", destination="JFK", flight_number="DL9", departure_time=f"{FLIGHT_DAY}T10:00:00")
    j = make_client().post("/copilot/options", json=b).json()
    assert j["alternatives"]["items"] == []
    assert "no_alternatives_found" in j["alternatives"]["warnings"]
    assert len(j["alternatives"]["search_handoffs"]) == 4
    assert j["refund_rights"]["applies"] is True


def test_non_us_route_per_state_column(make_client):
    b = body()
    b["flight"].update(origin="SJU", destination="BQN", airline="BA", flight_number="BA304")
    j = make_client().post("/copilot/options", json=b).json()
    assert j["refund_rights"] == {"applies": False, "jurisdiction": None, "content_key": None}
    assert j["alternatives"]["items"] == [] and "non_us_route" in j["alternatives"]["warnings"]
    assert j["alternatives"]["search_handoffs"]


def test_unknown_airline(make_client):
    b = body()
    b["flight"].update(airline="ZZ", flight_number="ZZ1")
    j = make_client().post("/copilot/options", json=b).json()
    assert j["airline"] is None and "unknown_airline" in j["warnings"]
    assert j["alternatives"]["items"]          # still returns alternatives


def test_max_alternatives_truncates_after_ranking(make_client):
    j = make_client().post("/copilot/options", json=body(max_alternatives=1)).json()
    assert [i["flight_number"] for i in j["alternatives"]["items"]] == ["AA117"]


def test_scorer_failure_and_missing_scorer(make_client):
    def boom(*a):
        raise RuntimeError("model not loaded")
    for score in (boom, None):
        r = make_client(score=score).post("/copilot/options", json=body())
        assert r.status_code == 200
        j = r.json()
        items = j["alternatives"]["items"]
        assert "risk_unavailable" in j["warnings"]
        assert all(i["delay_probability"] is None and i["risk_level"] is None for i in items)
        assert [i["dep_local"] for i in items] == sorted(i["dep_local"] for i in items)


def test_affiliate_ids_absent_then_present(make_client, monkeypatch):
    c = make_client()
    assert c.post("/copilot/options", json=body()).json()["disclosure"]["required"] is False
    monkeypatch.setenv("AFFILIATE_KAYAK_PARAMS", "aid=123")
    monkeypatch.setenv("AFFILIATE_UBER_PARAMS", "client_id=abc")
    j = c.post("/copilot/options", json=body()).json()
    assert j["disclosure"] == {"required": True}
    kayak = next(h for h in j["alternatives"]["search_handoffs"] if h["partner"] == "kayak")
    assert kayak["is_affiliate"] is True and "aid=123" in kayak["url"]
    assert next(h for h in j["alternatives"]["search_handoffs"] if h["partner"] == "google_flights")["is_affiliate"] is False
    assert any(x["is_affiliate"] for x in j["ride"]["links"])


def test_affiliate_only_on_hotel_sets_disclosure(make_client, monkeypatch):
    monkeypatch.setenv("AFFILIATE_STAY22_PARAMS", "aid=9")
    j = make_client().post("/copilot/options", json=body()).json()
    assert j["hotel"]["links"][0]["partner"] == "stay22" and j["disclosure"]["required"] is True


@pytest.mark.parametrize("patch", [
    {"state": "bogus"},
    {"max_alternatives": 0},
    {"max_alternatives": 16},
    {"rebooked": "tomorrow", "state": "rebooked"},
    {"rebooked": "today"},                      # rebooked only valid with state=rebooked
    {"flight": {"flight_number": "DL1"}},       # missing fields
])
def test_validation_422(make_client, patch):
    assert make_client().post("/copilot/options", json=body(**patch)).status_code == 422


def test_unknown_airport_and_bad_time_422(make_client):
    c = make_client()
    b = body()
    b["flight"]["origin"] = "XXX"
    r = c.post("/copilot/options", json=b)
    assert r.status_code == 422 and r.json()["detail"]["reason"] == "unknown_airport"
    b = body()
    b["flight"]["departure_time"] = "not-a-time"
    assert c.post("/copilot/options", json=b).json()["detail"]["reason"] == "invalid_departure_time"
    b = body()
    b["flight"]["destination"] = "JFK"
    assert c.post("/copilot/options", json=b).json()["detail"]["reason"] == "same_airport"


def test_home_airport_never_logged(make_client, caplog):
    caplog.set_level(logging.DEBUG)
    make_client().post("/copilot/options", json=body(home_airport="SEA"))
    assert "SEA" not in caplog.text


def test_status_endpoint_sources(make_client):
    flight = body()["flight"]
    r = make_client().post("/copilot/status", json={"flight": flight})
    assert r.json() == {"status": "unknown", "source": "none", "as_of": None, "detail": None}
    r = make_client(status_source=SwimLiveStateSource()).post("/copilot/status", json={"flight": flight})
    assert r.status_code == 200
    assert r.json() == {"status": "unknown", "source": "swim", "as_of": None, "detail": "source_not_configured"}
    assert make_client().post("/copilot/status", json={}).status_code == 422


def test_status_source_from_env(make_client, monkeypatch):
    monkeypatch.setenv("FLYRELY_COPILOT_STATUS_SOURCE", "swim")
    r = make_client().post("/copilot/status", json={"flight": body()["flight"]})
    assert r.status_code == 200 and r.json()["detail"] == "source_not_configured"


def test_performance_40_flights(make_client, tmp_path):
    import time
    rows = [f"AA,{n},JFK,LAX,{6 + (n - 100) // 4:02d}:{(n % 4) * 15:02d},23:00,AA,{n},127,5,2026-08" for n in range(100, 140)]
    p = tmp_path / "big.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write(HEADER + "\n".join(rows) + "\n")
    c = make_client(now=noon_utc(hour=0))
    c.app.state.copilot_schedule = BtsTypicalSchedule(p)
    c.post("/copilot/options", json=body())     # warm up
    t = time.perf_counter()
    j = c.post("/copilot/options", json=body(max_alternatives=15)).json()
    assert len(j["alternatives"]["items"]) == 15
    assert time.perf_counter() - t < 0.3


def test_airline_passthrough_and_spirit_notice(make_client):
    c = make_client()
    j = c.post("/copilot/options", json=body()).json()
    assert "verification" in j["airline"] and "other_channels" in j["airline"] and j["airline_notice"] is None
    b = body()
    b["flight"].update(airline="NK", flight_number="NK100")
    j = c.post("/copilot/options", json=b).json()
    assert j["airline"] is None and j["airline_notice"]["status_url"].startswith("https://")
    assert j["airline_notice"]["note"] and "unknown_airline" not in j["warnings"]
