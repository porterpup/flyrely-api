"""GET /flights/lookup on a small fixture schedule (not the real artifact), plus the real artifact's UA648."""
import gzip
from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import flight_lookup
import main
from copilot.schedule import BtsTypicalSchedule

HEADER = ("marketing_airline,marketing_flight_number,origin,dest,dep_time,arr_time,operating_airline,"
          "operating_flight_number,dow_mask,days_seen,schedule_as_of\n")
ROWS = [
    "UA,648,EWR,SFO,10:40,13:51,UA,648,127,55,2026-08",     # every day
    "UA,648,SFO,MSP,16:10,21:57,UA,648,89,19,2026-08",      # Mon, Thu, Fri, Sun
    "UA,648,STL,EWR,06:01,09:30,UA,648,127,31,2026-08",
    "UA,900,JFK,LAX,12:00,15:30,OO,5521,96,4,2026-08",      # Sat+Sun only, regional operator
    "DL,7,LAX,JFK,23:00,07:10,DL,7,127,9,2026-08",          # red-eye (arrives next day)
    "DL,50,JFK,LAX,07:15,10:40,DL,50,31,6,2026-08",         # two frequencies of one number
    "DL,50,JFK,LAX,18:15,21:40,DL,50,31,6,2026-08",
    "AS,11,SEA,OGG,08:00,10:30,AS,11,127,12,2026-08",
]


def next_weekday(weekday: int) -> str:
    d = date.today() + timedelta(days=1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d.isoformat()


@pytest.fixture
def client(tmp_path, monkeypatch):
    p = tmp_path / "route_schedule.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write(HEADER + "\n".join(ROWS) + "\n")
    monkeypatch.setattr(flight_lookup, "_schedule", BtsTypicalSchedule(p))
    monkeypatch.setattr(flight_lookup, "_limiter", None)
    monkeypatch.delenv("FLYRELY_LOOKUP_RATE_PER_MIN", raising=False)
    return TestClient(main.app)


@pytest.mark.parametrize("raw,airline,expect", [
    ("UA648", None, ("UA", "648")), ("ua 648", None, ("UA", "648")), ("UA0648", None, ("UA", "648")),
    ("648", "UA", ("UA", "648")), ("B6 12", None, ("B6", "12")), ("9E4632", None, ("9E", "4632")),
])
def test_parse_query(raw, airline, expect):
    assert flight_lookup.parse_query(raw, airline) == expect


@pytest.mark.parametrize("raw,airline", [("648", None), ("UA", None), ("UA12345", None), ("UA648; DROP", None),
                                         ("UA648", "DL"), ("648", "U!"), ("", None)])
def test_parse_query_rejects(raw, airline):
    with pytest.raises(ValueError):
        flight_lookup.parse_query(raw, airline)


def test_lookup_forms_equivalent(client):
    d = next_weekday(0)
    a = client.get("/flights/lookup", params={"flight": "UA648", "date": d}).json()
    b = client.get("/flights/lookup", params={"flight": "UA 648", "date": d}).json()
    c = client.get("/flights/lookup", params={"flight": "648", "airline": "ua", "date": d}).json()
    assert a == b == c
    assert a["flight"] == "UA648" and a["source"] == "bts_typical_schedule" and a["schedule_as_of"] == "2026-08"
    assert "as of 2026-08" in a["note"] and "seasonal" in a["note"]


def test_multi_leg_sorted_by_departure(client):
    body = client.get("/flights/lookup", params={"flight": "UA648", "date": next_weekday(0)}).json()
    legs = [(m["origin"], m["dest"]) for m in body["matches"]]
    assert legs == [("STL", "EWR"), ("EWR", "SFO"), ("SFO", "MSP")]      # Monday: all three legs
    first = body["matches"][1]
    assert first["dep_time"] == "10:40" and first["arr_time"] == "13:51" and first["days_seen"] == 55
    assert first["operating_airline"] == "UA" and first["operating_flight_number"] == "UA648"


def test_dow_mask_filter(client):
    # mask 89 = bits 0,3,4,6 = Mon, Thu, Fri, Sun
    thu = client.get("/flights/lookup", params={"flight": "UA648", "date": next_weekday(3)}).json()
    assert ("SFO", "MSP") in [(m["origin"], m["dest"]) for m in thu["matches"]]
    tue = client.get("/flights/lookup", params={"flight": "UA648", "date": next_weekday(1)}).json()
    assert ("SFO", "MSP") not in [(m["origin"], m["dest"]) for m in tue["matches"]]


def test_weekday_miss_is_404_with_hint(client):
    r = client.get("/flights/lookup", params={"flight": "UA900", "date": next_weekday(1)})   # weekend-only
    assert r.status_code == 404
    assert "not usually on that weekday" in r.json()["detail"]["message"]


def test_codeshare_operator_reported(client):
    body = client.get("/flights/lookup", params={"flight": "UA900", "date": next_weekday(5)}).json()
    m = body["matches"][0]
    assert body["flight"] == "UA900"
    assert m["operating_airline"] == "OO" and m["operating_flight_number"] == "OO5521"
    assert m["operating_airline_name"] == "SkyWest Airlines"


def test_multiple_frequencies(client):
    body = client.get("/flights/lookup", params={"flight": "DL50", "date": next_weekday(2)}).json()
    assert [m["dep_time"] for m in body["matches"]] == ["07:15", "18:15"]


def test_red_eye_flags_next_day(client):
    m = client.get("/flights/lookup", params={"flight": "DL7", "date": next_weekday(2)}).json()["matches"][0]
    assert m["arr_next_day"] is True


def test_spirit_is_defunct(client):
    r = client.get("/flights/lookup", params={"flight": "NK123", "date": next_weekday(2)})
    assert r.status_code == 410
    d = r.json()["detail"]
    assert d["reason"] == "airline_defunct" and "Spirit Airlines no longer operates" in d["message"]


def test_hawaiian_maps_to_alaska(client):
    body = client.get("/flights/lookup", params={"flight": "HA11", "date": next_weekday(2)}).json()
    assert body["flight"] == "AS11" and body["matches"][0]["origin"] == "SEA"
    assert "Hawaiian" in body["note"] and "Alaska" in body["note"]


def test_not_found_404(client):
    r = client.get("/flights/lookup", params={"flight": "UA9999", "date": next_weekday(2)})
    assert r.status_code == 404
    d = r.json()["detail"]
    assert d["reason"] == "not_found" and d["message"] == flight_lookup.NOT_FOUND_MESSAGE
    assert "US domestic flights only" in d["message"] and "enter the details manually" in d["message"]


@pytest.mark.parametrize("params", [
    {"flight": "UA648", "date": "not-a-date"}, {"flight": "UA648", "date": "2019-01-01"},
    {"flight": "UA648", "date": "2099-01-01"}, {"flight": "648", "date": next_weekday(0)},
    {"flight": "UA64888", "date": next_weekday(0)},
])
def test_invalid_input_422(client, params):
    assert client.get("/flights/lookup", params=params).status_code == 422


def test_rate_limit(client, monkeypatch):
    monkeypatch.setenv("FLYRELY_LOOKUP_RATE_PER_MIN", "2")
    codes = [client.get("/flights/lookup", params={"flight": "UA648", "date": next_weekday(0)}).status_code
             for _ in range(4)]
    assert codes[:1] == [200] and 429 in codes


def test_real_artifact_ua648_ewr_sfo():
    flight_lookup._schedule = None
    c = TestClient(main.app)
    body = c.get("/flights/lookup", params={"flight": "UA648", "date": next_weekday(0)}).json()
    assert ("EWR", "SFO") in [(m["origin"], m["dest"]) for m in body["matches"]]


# ---- GET /flights/search ----------------------------------------------------

def test_search_route_with_predictions_sorted(client):
    body = client.get("/flights/search", params={"origin": "jfk", "dest": "LAX", "date": next_weekday(2)}).json()
    deps = [f["dep_time"] for f in body["flights"]]
    assert deps == sorted(deps) and len(deps) == 2          # the two DL50 frequencies (Wed)
    f = body["flights"][0]
    assert f["flight"] == "DL50" and f["airline"] == "DL" and f["airline_name"] == "Delta Air Lines"
    assert 0 <= f["delay_probability"] <= 1 and f["risk_level"] in ("low", "medium", "high")
    assert "Based on recent schedules" in body["note"] and body["schedule_as_of"] == "2026-08"


def test_search_codeshare_scored_for_operator(client):
    body = client.get("/flights/search", params={"origin": "JFK", "dest": "LAX", "date": next_weekday(5)}).json()
    f = body["flights"][0]
    assert f["flight"] == "UA900" and f["operating_airline"] == "OO" and f["operating_flight_number"] == "OO5521"


def test_search_time_window(client):
    p = {"origin": "JFK", "dest": "LAX", "date": next_weekday(2)}
    assert [f["dep_time"] for f in client.get("/flights/search", params={**p, "after": "12:00"}).json()["flights"]] == ["18:15"]
    assert client.get("/flights/search", params={**p, "before": "06:00"}).status_code == 404


def test_search_404_and_validation(client):
    d = next_weekday(2)
    assert client.get("/flights/search", params={"origin": "AAA", "dest": "BBB", "date": d}).status_code == 404
    for bad in ({"origin": "JF", "dest": "LAX"}, {"origin": "JFK", "dest": "JFK"}, {"origin": "J1K", "dest": "LAX"},
                {"origin": "JFK", "dest": "LAX", "after": "25:00"}, {"origin": "JFK", "dest": "LAX", "date": "x"}):
        assert client.get("/flights/search", params={"date": d, **bad}).status_code == 422, bad


def test_search_one_model_call_and_fast(client, monkeypatch):
    calls = []
    real = main.model_service.predict_many
    monkeypatch.setattr(main.model_service, "predict_many", lambda rows: calls.append(len(rows)) or real(rows))
    import time as _t
    t0 = _t.perf_counter()
    assert client.get("/flights/search", params={"origin": "JFK", "dest": "LAX", "date": next_weekday(2)}).status_code == 200
    assert calls == [2] and _t.perf_counter() - t0 < 2


def test_search_503_when_model_unavailable(client, monkeypatch):
    monkeypatch.setattr(main.model_service, "model", None)
    r = client.get("/flights/search", params={"origin": "JFK", "dest": "LAX", "date": next_weekday(2)})
    assert r.status_code == 503 and r.json()["detail"]["reason"] == "model_unavailable"


def test_predict_many_matches_predict():
    feats = [main.build_features("JFK", "LAX", datetime(2026, 11, 15, h, 0), "AA") for h in (7, 18)]
    batch = main.model_service.predict_many(feats)
    single = [main.model_service.predict(f) for f in feats]
    assert [round(b[0], 9) for b in batch] == [round(s[0], 9) for s in single]


def test_real_search_ewr_sfo_30_flights_speed():
    flight_lookup._schedule = None
    import time as _t
    t0 = _t.perf_counter()
    r = TestClient(main.app).get("/flights/search", params={"origin": "LAX", "dest": "SFO", "date": next_weekday(1)})
    assert r.status_code == 200 and len(r.json()["flights"]) >= 20
    assert _t.perf_counter() - t0 < 3


def test_search_unknown_or_non_us_airport_message(client):
    d = next_weekday(2)
    r = client.get("/flights/search", params={"origin": "LHR", "dest": "JFK", "date": d})
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["reason"] == "unknown_airport" and "LHR" in detail["message"]
    assert "US domestic flights only" in detail["message"]


def test_search_rows_carry_risk_factors(client):
    body = client.get("/flights/search", params={"origin": "JFK", "dest": "LAX", "date": next_weekday(2)}).json()
    for f in body["flights"]:
        assert isinstance(f["risk_factors"], list)
    evening = [f for f in body["flights"] if f["dep_time"] == "18:15"][0]
    assert any("Evening flights" in r for r in evening["risk_factors"])
