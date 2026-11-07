"""aviationweather.gov METAR/TAF parsing and selection, from recorded fixtures (no network)."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

import aviation_weather as aw
import main

FIX = Path(__file__).parent / "fixtures"
METARS = {r["icaoId"]: r for r in json.loads((FIX / "aviationweather_metar.json").read_text())}
TAFS = {r["icaoId"]: r for r in json.loads((FIX / "aviationweather_taf.json").read_text())}


def utc(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


# JFK TAF: 0712/0818 ... FM071800 28012G19KT ... FM072200 29017G25KT; TEMPO 0806/0810 BKN035
FM1800 = 1794074400
FM2200 = 1794088800


def test_visibility_parsing():
    assert aw.parse_visibility("10+") == 10
    assert aw.parse_visibility("6+") == 6
    assert aw.parse_visibility(3) == 3
    assert aw.parse_visibility("1/2") == 0.5
    assert aw.parse_visibility("1 1/2") == 1.5
    assert aw.parse_visibility("P6SM") == 6
    assert aw.parse_visibility("") is None
    assert aw.parse_visibility(None) is None
    assert aw.parse_visibility("garbage") is None


def test_metar_current_conversion():
    w = aw.compose_weather(METARS["KJFK"], TAFS["KJFK"], None)  # no time: current
    assert w["source"] == "current"
    assert w["temperature_f"] == pytest.approx(METARS["KJFK"]["temp"] * 9 / 5 + 32, abs=0.1)  # 14.4C -> 57.9F
    assert w["wind_speed_mph"] == pytest.approx(METARS["KJFK"]["wspd"] * 1.15078, abs=0.1)     # knots -> mph
    assert w["visibility_miles"] == 10  # "10+"
    assert w["as_of"].endswith("Z")
    assert w["conditions"]


def test_taf_group_selection_uses_forecast_in_range():
    w = aw.compose_weather(METARS["KJFK"], TAFS["KJFK"], utc(FM1800 + 3600))
    assert w["source"] == "forecast"
    assert w["wind_speed_mph"] == pytest.approx(12 * 1.15078, abs=0.1)  # FM071800 group: 12 kt
    w2 = aw.compose_weather(METARS["KJFK"], TAFS["KJFK"], utc(FM2200 + 60))
    assert w2["wind_speed_mph"] == pytest.approx(17 * 1.15078, abs=0.1)  # next FM group
    assert w2["as_of"] == utc(FM2200 + 60).strftime("%Y-%m-%dT%H:%M:%SZ")
    # forecast temperature is the observed one: TAFs carry none
    assert w2["temperature_f"] == aw.compose_weather(METARS["KJFK"], None, None)["temperature_f"]


def test_initial_group_before_first_fm():
    w = aw.compose_weather(METARS["KJFK"], TAFS["KJFK"], utc(TAFS["KJFK"]["validTimeFrom"] + 60))
    assert w["source"] == "forecast"
    assert w["wind_speed_mph"] == pytest.approx(10 * 1.15078, abs=0.1)


def test_outside_taf_period_falls_back_to_current():
    for t in (TAFS["KJFK"]["validTimeTo"] + 3600, TAFS["KJFK"]["validTimeFrom"] - 3600):
        w = aw.compose_weather(METARS["KJFK"], TAFS["KJFK"], utc(t))
        assert w["source"] == "current"


def test_tempo_overlay_merged_worst_case():
    # JFK TEMPO 0806/0810 BKN035 -> a ceiling appears inside the window only
    tempo = next(g for g in TAFS["KJFK"]["fcsts"] if g["fcstChange"] == "TEMPO")
    inside = aw.compose_weather(METARS["KJFK"], TAFS["KJFK"], utc(tempo["timeFrom"] + 600))
    outside = aw.compose_weather(METARS["KJFK"], TAFS["KJFK"], utc(tempo["timeTo"] + 600))
    assert inside["ceiling_ft"] == 3500
    assert outside["ceiling_ft"] is None


def test_tempo_thunderstorm_in_san_juan_taf():
    # TJSJ TEMPO 0717/0721 2SM +TSRA ... OVC040
    tempo = next(g for g in TAFS["TJSJ"]["fcsts"] if g["fcstChange"] == "TEMPO")
    w = aw.compose_weather(METARS["TJSJ"], TAFS["TJSJ"], utc(tempo["timeFrom"] + 600))
    assert w["source"] == "forecast"
    assert w["visibility_miles"] == 2
    assert "thunderstorm" in w["conditions"]
    assert any("Thunderstorms" in f for f in aw.weather_risk_factors(w, "destination"))


def test_missing_fields_are_handled():
    assert aw.compose_weather(None, TAFS["KJFK"], utc(FM1800)) is None
    assert aw.compose_weather({"icaoId": "KJFK"}, None, None) is None  # no temp
    bare = {"temp": 10, "wspd": None, "visib": None}
    assert aw.compose_weather(bare, None, None) is None
    # TAF group with no wind/visibility falls back to the METAR values
    taf = {"validTimeFrom": 0, "validTimeTo": 10**10,
           "fcsts": [{"timeFrom": 0, "timeTo": 10**10, "fcstChange": None, "wspd": None, "visib": "", "clouds": []}]}
    w = aw.compose_weather(METARS["KJFK"], taf, utc(1794074400))
    assert w["source"] == "forecast" and w["visibility_miles"] == 10
    assert aw.select_taf_group({}, 1) is None
    assert aw.select_taf_group({"validTimeFrom": 0, "validTimeTo": 10, "fcsts": None}, 5) is None


def test_risk_factor_text_ts_wintry_ceiling():
    f = aw.weather_risk_factors({"source": "forecast", "wx": "+TSRA", "ceiling_ft": 800}, "origin")
    assert any("Thunderstorms in the forecast" in x and "origin" in x for x in f)
    assert any("Low cloud ceiling (about 800 ft)" in x for x in f)
    assert all("may be more likely" in x for x in f)
    snow = aw.weather_risk_factors({"source": "current", "wx": "-SN BR", "ceiling_ft": None}, "destination")
    assert snow == ["Snow or freezing precipitation reported now at destination; delays may be more likely"]
    assert aw.weather_risk_factors({"source": "current", "wx": "-FZRA"}, "origin")
    assert aw.weather_risk_factors({"source": "current", "wx": "RA BR", "ceiling_ft": 1500}, "origin") == []
    assert aw.weather_risk_factors({"ceiling_ft": 1000}, "origin") == []  # not below 1,000 ft


def test_get_risk_factors_includes_weather():
    bad = {"temperature_f": 50, "wind_speed_mph": 5, "visibility_miles": 10, "conditions": "x",
           "source": "forecast", "wx": "TS", "ceiling_ft": 500}
    factors = main.get_risk_factors({"dep_hour": 10}, bad, None)
    assert any("Thunderstorms" in x for x in factors)
    assert any("Low cloud ceiling" in x for x in factors)


# ---- client: batching, caches, headers, failure ---------------------------------------------------------

class FakeAPI:
    def __init__(self, fail=False):
        self.requests = []
        self.fail = fail

    def handler(self, request: httpx.Request):
        self.requests.append(request)
        if self.fail:
            raise httpx.ConnectError("down")
        product = request.url.path.rsplit("/", 1)[-1]
        ids = request.url.params["ids"].split(",")
        data = {"metar": METARS, "taf": TAFS}[product]
        return httpx.Response(200, json=[data[i] for i in ids if i in data])


def make_client(api):
    http = httpx.AsyncClient(transport=httpx.MockTransport(api.handler), headers={"User-Agent": aw.USER_AGENT})
    return aw.AviationWeatherClient(http)


def test_client_batches_and_caches(monkeypatch):
    api = FakeAPI()
    c = make_client(api)
    when = utc(FM1800 + 600)
    out = asyncio.run(c.get_many([("KJFK", when), ("KLAX", when)]))
    assert out[0]["source"] == "forecast" and out[1]["source"] == "forecast"
    assert len(api.requests) == 2  # one METAR + one TAF request for both airports
    assert {r.url.params["ids"] for r in api.requests} == {"KJFK,KLAX"}
    assert all(r.headers["user-agent"] == "FlyRely/1.0" for r in api.requests)
    asyncio.run(c.get_many([("KJFK", when)]))
    assert len(api.requests) == 2  # cache hit

    # METAR expires after 10 min, TAF only after 30
    real = aw.time.monotonic
    monkeypatch.setattr(aw.time, "monotonic", lambda: real() + 11 * 60)
    asyncio.run(c.get_many([("KJFK", when)]))
    assert [r.url.path.rsplit("/", 1)[-1] for r in api.requests[2:]] == ["metar"]
    monkeypatch.setattr(aw.time, "monotonic", lambda: real() + 31 * 60)
    asyncio.run(c.get_many([("KJFK", when)]))
    assert sorted(r.url.path.rsplit("/", 1)[-1] for r in api.requests[3:]) == ["metar", "taf"]


def test_client_provider_down_returns_none():
    api = FakeAPI(fail=True)
    out = asyncio.run(make_client(api).get_many([("KJFK", utc(FM1800)), ("KLAX", None)]))
    assert out == [None, None]


# ---- service wiring ---------------------------------------------------------------------------------------

def test_provider_defaults(monkeypatch):
    assert main.WeatherService("").provider == "aviationweather"
    assert main.WeatherService("key").provider == "tomorrow"
    assert main.WeatherService("key", "openweathermap").provider == "openweathermap"
    assert main.WeatherService("", "tomorrow").provider == "aviationweather"  # no key: free provider
    assert main.WeatherService("", "none").provider == "none"
    assert main.WeatherService("", "").cost_per_call == 0
    assert main.WeatherService("key").cost_per_call == main.WEATHER_COST_PER_CALL


def test_service_maps_non_k_airports_to_icao():
    svc = main.WeatherService("")
    assert svc.icao("JFK") == "KJFK"
    assert svc.icao("ANC") == "PANC"
    assert svc.icao("HNL") == "PHNL"
    assert svc.icao("SJU") == "TJSJ"


def test_service_pair_uses_icao_ids():
    api = FakeAPI()
    svc = main.WeatherService("")
    svc.aviation = make_client(api)
    when = utc(FM1800 + 600)
    jfk, sju = asyncio.run(svc.get_weather_pair([("JFK", when), ("SJU", when)]))
    assert jfk and sju
    assert api.requests[0].url.params["ids"] == "KJFK,TJSJ"


def test_predict_succeeds_when_weather_provider_down(monkeypatch):
    from fastapi.testclient import TestClient
    svc = main.WeatherService("")
    svc.aviation = make_client(FakeAPI(fail=True))
    monkeypatch.setattr(main, "weather_service", svc)
    r = TestClient(main.app).post("/predict", json={
        "origin": "JFK", "destination": "LAX", "departure_time": "2026-11-15T14:30:00", "airline": "DL"})
    assert r.status_code == 200
    body = r.json()
    assert body["weather_available"] is False and body["origin_weather"] is None


def test_predict_with_weather_fixture(monkeypatch):
    from fastapi.testclient import TestClient
    svc = main.WeatherService("")
    svc.aviation = make_client(FakeAPI())
    monkeypatch.setattr(main, "weather_service", svc)
    # 2026-11-08 08:00 New York local = 13:00Z, inside the fixture TAFs' validity (0712 - 0818)
    r = TestClient(main.app).post("/predict", json={
        "origin": "JFK", "destination": "LAX", "departure_time": "2026-11-08T08:00:00", "airline": "DL"})
    body = r.json()
    assert r.status_code == 200 and body["weather_available"] is True
    assert body["origin_weather"]["source"] == "forecast"
    assert set(body["origin_weather"]) >= {"temperature_f", "wind_speed_mph", "visibility_miles", "conditions"}


def test_weather_times_use_origin_timezone():
    dep, arr = main._weather_times("JFK", "LAX", datetime(2026, 11, 8, 8, 0))
    assert dep == datetime(2026, 11, 8, 13, 0, tzinfo=timezone.utc)  # EST = UTC-5
    assert timedelta(hours=5) < arr - dep < timedelta(hours=7)
    assert main._weather_times("ZZZ", "LAX", datetime(2026, 11, 8, 8, 0)) == (None, None)


def test_usage_note_free_provider(monkeypatch):
    svc = main.WeatherService("")
    assert svc.cost_per_call == 0
