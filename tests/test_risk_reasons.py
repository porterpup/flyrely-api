"""Risk-factor reasons: time of day and busy airports, tied to the model's own features."""
from datetime import datetime

import main


def factors(hour, origin="JFK", dest="LAX", **kw):
    f = main.build_features(origin, dest, datetime(2026, 11, 18, hour), "DL")
    f.update(kw)
    return main.get_risk_factors(f, None, None, origin, dest)


def test_early_morning_reason():
    assert "Early-morning flights are rarely delayed" in factors(6)


def test_evening_reason():
    assert any(r.startswith("Evening flights are more often delayed") for r in factors(19))
    assert not any("Early-morning" in r for r in factors(19))


def test_midday_and_night_have_no_time_reason():
    for h in (10, 23, 2):
        assert main.time_of_day_reason(h) is None


def test_busy_airport_reason_names_airports():
    both = [r for r in factors(10) if "busy" in r]
    assert both == ["JFK and LAX are busy airports, where delays spread more easily"]
    one = [r for r in factors(10, "FWA", "JFK") if "busy" in r]
    assert one == ["JFK is a busy airport, where delays spread more easily"]
    assert not [r for r in factors(10, "FWA", "ROA") if "busy" in r]


def test_weather_reasons_come_before_time_of_day():
    f = main.build_features("JFK", "LAX", datetime(2026, 11, 18, 19), "DL")
    wx = {"temperature_f": 50, "wind_speed_mph": 30, "visibility_miles": 10, "conditions": "clear"}
    out = main.get_risk_factors(f, wx, None, "JFK", "LAX")
    assert "wind" in out[0].lower()


def test_predict_returns_reasons():
    from fastapi.testclient import TestClient
    r = TestClient(main.app).post("/predict", json={"origin": "JFK", "destination": "LAX",
                                                     "departure_time": "2026-11-18T19:00:00", "airline": "DL"})
    assert r.status_code == 200
    assert any("Evening flights" in x for x in r.json()["risk_factors"])
