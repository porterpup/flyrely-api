"""T6: POST /predict/itinerary (feature-flagged) on a synthetic ladder bundle and a fixture schedule."""
import gzip
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import arrival_fakes
import arrival_service
import flight_lookup
import itinerary
import main
from copilot.schedule import BtsTypicalSchedule

ROOT = Path(__file__).resolve().parent.parent
HEADER = ("marketing_airline,marketing_flight_number,origin,dest,dep_time,arr_time,operating_airline,"
          "operating_flight_number,dow_mask,days_seen,schedule_as_of\n")
ROWS = ["UA,648,SFO,DEN,07:10,10:42,UA,648,127,55,2026-08",
        "DL,1999,ATL,BHM,09:00,08:58,DL,1999,127,55,2026-08",     # westbound short hop, same day (ET -> CT)
        "DL,2400,ATL,BNA,23:55,23:58,DL,2400,127,55,2026-08",     # late westbound, same day (00:58 ET)
        "UA,1700,SFO,EWR,23:00,07:25,UA,1700,127,55,2026-08"]     # eastbound red-eye, next day
DAY = "2030-12-03"


def leg(number, o, d, dep, arr=None, airline=None, **kw):
    out = {"flight_id": f"f-{number}", "flight_number": number, "airline": airline or number[:2],
           "origin": o, "destination": d, "departure_time": f"{DAY}T{dep}:00"}
    if arr:
        out["arrival_time"] = f"{DAY}T{arr}:00"
    return {**out, **kw}


UA648 = leg("UA648", "SFO", "DEN", "07:10", "10:42")
UA1442 = leg("UA1442", "DEN", "BNA", "11:35", "15:05")          # 53 min layover at DEN


@pytest.fixture
def seen():
    return []


@pytest.fixture
def client(monkeypatch, tmp_path, seen):
    monkeypatch.setattr(main, "arrival_service", arrival_fakes.make_service(seen))
    p = tmp_path / "route_schedule.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write(HEADER + "\n".join(ROWS) + "\n")
    monkeypatch.setattr(flight_lookup, "_schedule", BtsTypicalSchedule(p))
    app = FastAPI()
    itinerary.mount(app, main)
    return TestClient(app)


def post(client, legs, connections=None, **kw):
    body = {"itinerary_id": "trip-123", "legs": legs, **kw}
    if connections is not None:
        body["connections"] = connections
    return client.post("/predict/itinerary", json=body)


def reason(r):
    return r.json()["detail"]["reason"]


# ------------------------------------------------------------------ happy path and the app's request shape

def test_two_leg_connection(client):
    r = post(client, [UA648, UA1442], [{}])
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["itinerary_id"] == "trip-123"
    assert [l["flight_id"] for l in b["legs"]] == ["f-UA648", "f-UA1442"]
    assert all(l["arrival"]["model_version"] == "arr-ladder-v1" for l in b["legs"])   # full /predict per leg
    c = b["connections"][0]
    assert (c["at"], c["kind"], c["scheduled_layover_minutes"]) == ("DEN", "connection", 53)
    assert (c["min_connect_minutes"], c["mct_source"]) == (50, "default_large_campus")
    assert c["validated"] is True and c["not_validated_reason"] is None
    assert 0 < c["miss_probability"] < 1 and c["risk_level"] in ("low", "medium", "high")
    assert 0 < b["trip"]["disruption_probability"] < 1 and b["trip"]["approximate"] is False
    assert b["model_version"] == "itin-v1 (arr-ladder-v1, dep-copula-R1)"


def test_accepts_the_apps_trip_to_itinerary_request_shape(client):
    """flyrely-app src/lib/tripGrouping.ts tripToItineraryRequest (feat/trips): arrival_time may be undefined (key
    omitted) on the last leg, and `connections` has `{}` or `{min_connect_minutes}` per connection."""
    legs = [{k: v for k, v in l.items()} for l in (UA648, UA1442)]
    del legs[1]["arrival_time"]
    r = post(client, legs, [{}])
    assert r.status_code == 200, r.text
    r = post(client, legs, [{"min_connect_minutes": 60}])
    assert r.status_code == 200


def test_mct_override_is_echoed_and_below_minimum_is_not_validated(client):
    c = post(client, [UA648, UA1442], [{"min_connect_minutes": 60}]).json()["connections"][0]
    assert (c["min_connect_minutes"], c["mct_source"]) == (60, "user")
    assert c["validated"] is False and c["not_validated_reason"] == "layover_below_minimum"
    assert c["miss_probability"] is None and c["risk_level"] is None
    assert "Layover is shorter than the usual minimum at DEN" in c["warnings"]
    ok = post(client, [UA648, UA1442], [{"min_connect_minutes": 40}]).json()["connections"][0]
    assert (ok["min_connect_minutes"], ok["mct_source"], ok["validated"]) == (40, "user", True)


def test_longer_layover_has_lower_miss_probability(client):
    late = leg("UA1442", "DEN", "BNA", "13:35", "17:05")
    a = post(client, [UA648, UA1442], [{}]).json()["connections"][0]["miss_probability"]
    b = post(client, [UA648, late], [{}]).json()["connections"][0]["miss_probability"]
    assert b < a


def test_arrival_time_resolved_from_the_typical_schedule(client):
    no_arr = {k: v for k, v in UA648.items() if k != "arrival_time"}
    c = post(client, [no_arr, UA1442], [{}])
    assert c.status_code == 200, c.text
    assert c.json()["connections"][0]["scheduled_layover_minutes"] == 53     # schedule says 10:42


def test_three_legs_are_approximate(client):
    third = leg("UA77", "BNA", "ORD", "17:30", "18:40")
    b = post(client, [UA648, UA1442, third], [{}, {}]).json()
    assert b["trip"]["approximate"] is True and len(b["connections"]) == 2


def test_single_leg_has_no_connections(client):
    b = post(client, [UA648]).json()
    assert b["connections"] == [] and b["trip"]["disruption_probability"] == pytest.approx(
        arrival_fakes.CANCEL_P, abs=1e-3)


# ------------------------------------------------------------------ 422 reasons

def test_not_connected(client):
    r = post(client, [UA648, leg("UA1442", "SEA", "BNA", "11:35", "15:05")], [{}])
    assert r.status_code == 422 and reason(r) == "not_connected"


def test_schedule_unknown(client):
    nope = leg("UA999", "SFO", "DEN", "07:10")
    r = post(client, [nope, UA1442], [{}])
    assert r.status_code == 422 and reason(r) == "schedule_unknown"


def test_more_than_four_legs(client):
    legs = [leg("UA1", "SFO", "DEN", "01:00", "02:00"), leg("UA2", "DEN", "ORD", "03:00", "04:00"),
            leg("UA3", "ORD", "BNA", "05:00", "06:00"), leg("UA4", "BNA", "ATL", "07:00", "08:00"),
            leg("UA5", "ATL", "MCO", "09:00", "10:00")]
    r = post(client, legs, [{}] * 4)
    assert r.status_code == 422 and reason(r) == "too_many_legs"


@pytest.mark.parametrize("second_dep", ["10:42", "10:00", "11:00-next"])
def test_invalid_layover(client, second_dep):
    nxt = dict(UA1442)
    if second_dep.endswith("-next"):
        nxt["departure_time"] = "2030-12-05T11:00:00"        # more than 24 h after landing
    else:
        nxt["departure_time"] = f"{DAY}T{second_dep}:00"     # 0 min / negative
    r = post(client, [UA648, nxt], [{}])
    assert r.status_code == 422 and reason(r) == "invalid_layover"


def test_connections_length_mismatch(client):
    r = post(client, [UA648, UA1442], [{}, {}])
    assert r.status_code == 422 and reason(r) == "invalid_connections"


# ------------------------------------------------------------------ validated=false cases and through flights

def test_through_flight(client):
    same = leg("UA648", "DEN", "BNA", "11:35", "15:05", flight_id="f-b")
    b = post(client, [UA648, same], [{}]).json()
    c = b["connections"][0]
    assert c["kind"] == "through_flight" and c["miss_probability"] is None and c["validated"] is False
    assert c["not_validated_reason"] == "through_flight"
    assert any("Same flight number continues" in w for w in c["warnings"])
    assert b["trip"]["disruption_probability"] is not None       # same plane: no miss, cancellations only


def test_different_networks_hide_the_percent_and_warn(client):
    dl = leg("DL1442", "DEN", "BNA", "11:35", "15:05")
    b = post(client, [UA648, dl], [{}]).json()
    c = b["connections"][0]
    assert c["validated"] is False and c["not_validated_reason"] == "different_airlines"
    assert c["miss_probability"] is None and c["risk_level"] is None
    assert "Separate airlines: if your first flight is late, the second won't wait. Allow at least 2–3 hours." \
        in c["warnings"]
    assert c["mct_source"].endswith("+cross_network") and c["min_connect_minutes"] == 80   # 50 + 30
    assert b["trip"]["disruption_probability"] is None


def test_hawaiian_and_alaska_are_one_network(client):
    b = post(client, [leg("AS11", "SEA", "OGG", "08:00", "10:30"),
                      leg("HA45", "OGG", "HNL", "11:40", "12:30")], [{}]).json()
    c = b["connections"][0]
    assert c["not_validated_reason"] != "different_airlines"


def test_layover_over_four_hours_not_validated(client):
    late = leg("UA1442", "DEN", "BNA", "16:00", "19:00")          # 5 h 18 min
    c = post(client, [UA648, late], [{}]).json()["connections"][0]
    assert c["validated"] is False and c["not_validated_reason"] == "layover_over_4h"
    assert c["miss_probability"] is None


def test_hub_outside_top30_not_validated(client):
    legs = [leg("UA1", "SFO", "OKC", "07:00", "12:00"), leg("UA2", "OKC", "BNA", "13:30", "15:00")]
    c = post(client, legs, [{}]).json()["connections"][0]
    assert c["validated"] is False and c["not_validated_reason"] == "airport_not_tested"


# ------------------------------------------------------------------ 503s

def test_503_when_ladder_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "arrival_service", arrival_service.ArrivalService(ladder_path=tmp_path / "x.joblib"))
    r = post(client, [UA648, UA1442], [{}])
    assert r.status_code == 503 and reason(r) == "model_unavailable"
    # /predict keeps working
    assert TestClient(main.app).post("/predict", json={"origin": "SFO", "destination": "DEN",
                                                        "departure_time": f"{DAY}T07:10:00", "airline": "UA"}
                                     ).status_code == 200


def test_503_when_dependence_file_missing(client, monkeypatch):
    monkeypatch.setattr(main, "arrival_service", arrival_fakes.make_service(dependence=False))
    r = post(client, [UA648, UA1442], [{}])
    assert r.status_code == 503 and reason(r) == "model_unavailable"


def test_503_when_a_leg_fails(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(main.arrival_service.provider, "_feature_fn", boom)
    r = post(client, [UA648, UA1442], [{}])
    assert r.status_code == 503 and reason(r) == "prediction_failed"


# ------------------------------------------------------------------ codeshare, logging

def test_operating_carrier_is_used_for_the_model(client, seen):
    ua = leg("UA5521", "DEN", "LAX", "12:00", "13:30")
    first = leg("UA648", "SFO", "DEN", "07:10", "10:42")
    b = post(client, [first, ua], [{}]).json()
    assert b["legs"][1]["operating_airline"] == "OO"
    assert seen[-1] == "OO"


def test_logs_per_leg_and_per_connection(client):
    store = main.prediction_store
    a, b = dict(UA648, flight_id="log-a"), dict(UA1442, flight_id="log-b")
    r = post(client, [a, b], [{}], itinerary_id="trip-log-1")
    assert r.status_code == 200
    conn = sqlite3.connect(store.sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        legs = [dict(x) for x in conn.execute(
            "SELECT * FROM predictions WHERE flight_id IN ('log-a','log-b') ORDER BY flight_id")]
        its = [dict(x) for x in conn.execute("SELECT * FROM itinerary_predictions WHERE itinerary_id='trip-log-1'")]
    finally:
        conn.close()
    assert len(legs) == 2 and all(abs(x["arr_probability"] - arrival_fakes.ARR_P[3]) < 1e-3 for x in legs)
    assert legs[0]["arr_model_version"] == "arr-ladder-v1" and legs[0]["cancel_probability"] > 0
    (row,) = its
    assert row["at"] == "DEN" and row["mct_min"] == 50 and row["validated"] == 1
    assert json.loads(row["leg_flight_ids"]) == ["log-a", "log-b"]
    assert row["model_version"].startswith("itin-v1")


# ------------------------------------------------------------------ flag and CORS (fresh interpreter)

def _run(code, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("FLYRELY_FEATURE")}
    e.update(env)
    e["PYTHONPATH"] = str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e, cwd=ROOT, timeout=180)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


PROBE = (
    "import json, sys, main\n"
    "from fastapi.testclient import TestClient\n"
    "c = TestClient(main.app)\n"
    "hdr = {'Origin': 'http://localhost:5173', 'Access-Control-Request-Method': 'POST'}\n"
    "bad = {'Origin': 'https://evil.example', 'Access-Control-Request-Method': 'POST'}\n"
    "print(json.dumps({'paths': sorted(r.path for r in main.app.routes),\n"
    " 'post': c.post('/predict/itinerary', json={'legs': [{'origin': 'SFO', 'destination': 'DEN', 'departure_time': '2030-12-03T07:10:00'}]}).status_code,\n"
    " 'imported': 'itinerary' in sys.modules,\n"
    " 'ok': c.options('/predict', headers=hdr).headers.get('access-control-allow-origin'),\n"
    " 'ok_itin': c.options('/predict/itinerary', headers=hdr).headers.get('access-control-allow-origin'),\n"
    " 'bad': c.options('/predict', headers=bad).headers.get('access-control-allow-origin'),\n"
    " 'bad_itin': c.options('/predict/itinerary', headers=bad).headers.get('access-control-allow-origin')}))\n"
)


def test_route_absent_when_flag_off():
    off = _run(PROBE)
    assert "/predict/itinerary" not in off["paths"]
    assert off["post"] == 404 and off["imported"] is False
    assert _run(PROBE, FLYRELY_FEATURE_ITINERARY="true")["paths"] == off["paths"]   # only "1" turns it on


def test_flag_on_mounts_route_and_cors_is_unchanged():
    off = _run(PROBE)
    on = _run(PROBE, FLYRELY_FEATURE_ITINERARY="1")
    assert "/predict/itinerary" in on["paths"]
    assert on["post"] == 503            # ladder file not on this machine / CI: the whole call is a clean 503
    assert off["ok"] == on["ok"] == on["ok_itin"] == "http://localhost:5173"
    assert off["bad"] is None and on["bad"] is None and on["bad_itin"] is None


# ------------------------------------------------------------------ T5 dependence file: recalibration and outcome guard

def _dep_file(tmp_path, **extra):
    p = tmp_path / "connection_dependence_v1.json"
    p.write_text(json.dumps({"variant": "R1", "global_rho": 0.3, "by_cell": {}, "shrink_n": 5000.0, **extra}))
    return p


def _service_with_file(path):
    """The fake ladder plus the dependence fields exactly as ArrivalService._load reads them from `path`."""
    real = arrival_service.ArrivalService(ladder_path=Path("/nonexistent"), dependence_path=path)
    svc = arrival_fakes.make_service()
    svc.dependence, svc.dependence_error = real.dependence, real.dependence_error
    svc.recalibration, svc.decision_outcome = real.recalibration, real.decision_outcome
    return svc


def test_isotonic_recalibration_matches_sklearn():
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(1)
    p = rng.random(500)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, (rng.random(500) < p ** 1.5).astype(float))
    recal = arrival_service.load_recalibration({"recalibration": {"type": "isotonic", "x": iso.X_thresholds_.tolist(),
                                                                  "y": iso.y_thresholds_.tolist()}})
    q = np.array([-0.1, 0.0, 0.03, 0.2, 0.5, 0.77, 1.0, 1.2])
    got = [arrival_service.apply_recalibration(v, recal) for v in q]
    assert np.allclose(got, iso.predict(q))


def test_recalibration_is_applied_to_the_miss_probability(client, monkeypatch, tmp_path):
    raw = post(client, [UA648, UA1442], [{}]).json()["connections"][0]["miss_probability"]
    svc = _service_with_file(_dep_file(tmp_path, recalibration={"type": "isotonic", "x": [0.0, 1.0], "y": [0.5, 0.5]},
                                       chosen="R1+iso", decision_outcome=arrival_service.SHIP_OUTCOME))
    assert svc.dependence is not None, svc.dependence_error
    monkeypatch.setattr(main, "arrival_service", svc)
    c = post(client, [UA648, UA1442], [{}]).json()["connections"][0]
    assert raw != 0.5 and c["miss_probability"] == 0.5 and c["risk_level"] == "high"


def test_null_recalibration_leaves_the_copula_number(client, monkeypatch, tmp_path):
    raw = post(client, [UA648, UA1442], [{}]).json()["connections"][0]["miss_probability"]
    svc = _service_with_file(_dep_file(tmp_path, recalibration=None, chosen="R1",
                                       decision_outcome=arrival_service.SHIP_OUTCOME))
    monkeypatch.setattr(main, "arrival_service", svc)
    assert post(client, [UA648, UA1442], [{}]).json()["connections"][0]["miss_probability"] == raw


@pytest.mark.parametrize("outcome", ["serve_layover_table_B-L_itin-layover-table-v1", "no_connection_numbers",
                                     "no_connection_numbers_c6_failed", None])
def test_503_when_the_evaluation_did_not_ship_copula_numbers(client, monkeypatch, tmp_path, outcome):
    """Fail closed: any outcome but ship, or a file with no decision_outcome at all, is refused."""
    extra = {} if outcome is None else {"decision_outcome": outcome}
    svc = _service_with_file(_dep_file(tmp_path, recalibration=None, chosen="R1", **extra))
    assert svc.dependence is None and repr(outcome) in svc.dependence_error
    monkeypatch.setattr(main, "arrival_service", svc)
    r = post(client, [UA648, UA1442], [{}])
    assert r.status_code == 503 and reason(r) == "model_unavailable"


@pytest.mark.parametrize("recal", [{"type": "platt", "a": 1}, {"type": "isotonic", "x": [0.0, 1.0], "y": [0.2]},
                                   {"type": "isotonic", "x": [0.5, 0.1], "y": [0.1, 0.2]},
                                   {"type": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.5]}])
def test_malformed_recalibration_refuses_the_file(tmp_path, recal):
    svc = _service_with_file(_dep_file(tmp_path, recalibration=recal, decision_outcome=arrival_service.SHIP_OUTCOME))
    assert svc.dependence is None and svc.recalibration is None and svc.dependence_error


# ------------------------------------------------------------------ schedule-derived arrival: zones and midnight

def test_westbound_short_hop_lands_the_same_day(client):
    """ATL 09:00 ET -> BHM 08:58 CT: earlier on the clock, but 58 min later in real time."""
    r = post(client, [leg("DL1999", "ATL", "BHM", "09:00"), leg("DL2000", "BHM", "IAH", "10:00", "12:00")], [{}])
    assert r.status_code == 200, r.text
    assert r.json()["connections"][0]["scheduled_layover_minutes"] == 62


def test_eastbound_red_eye_lands_the_next_day(client):
    nxt = {**leg("UA1800", "EWR", "BOS", "08:30", "09:45"), "departure_time": "2030-12-04T08:30:00"}
    r = post(client, [leg("UA1700", "SFO", "EWR", "23:00"), nxt], [{}])
    assert r.status_code == 200, r.text
    assert r.json()["connections"][0]["scheduled_layover_minutes"] == 65


def test_departure_match_wraps_across_midnight(client):
    """Entered as 00:05 on the 4th; the schedule has 23:55 on the 3rd (10 min away), which then dates the arrival."""
    first = {**leg("DL2400", "ATL", "BNA", "00:05"), "departure_time": "2030-12-04T00:05:00"}
    nxt = {**leg("DL2401", "BNA", "MEM", "01:00", "02:00"), "departure_time": "2030-12-04T01:00:00"}
    r = post(client, [first, nxt], [{}])
    assert r.status_code == 200, r.text
    assert r.json()["connections"][0]["scheduled_layover_minutes"] == 62      # 23:58 CT on the 3rd -> 01:00
