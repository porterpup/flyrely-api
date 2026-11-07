"""models/airports_ref.csv: coverage, time-zone agreement with the old h2h table, /predict behaviour."""

import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

import main

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "headtohead"))
import h2h  # noqa: E402

client = TestClient(main.app)
REQ = {"origin": "FWA", "destination": "ATL", "departure_time": "2026-11-15T18:30:00", "airline": "DL"}

# The hand-written table h2h used before it read the CSV (kept here to prove agreement)
OLD_TZ = {
    "America/New_York": "ATL BOS BWI CLT CVG DCA DTW EWR FLL IAD JAX JFK LGA MCO MIA ORF PBI PHL PIT "
                        "RDU RIC RSW SAV SRQ SYR TPA BDL BUF CHS CLE CMH IND PVD ALB GSP MYR PWM BTV",
    "America/Chicago": "AUS BNA DAL DFW HOU IAH MCI MDW MEM MKE MSP MSY OKC OMA ORD SAT STL TUL DSM BHM LIT ICT MSN",
    "America/Denver": "DEN ABQ BOI SLC ELP COS", "America/Phoenix": "PHX TUS",
    "America/Los_Angeles": "BUR LAS LAX LGB OAK ONT PDX SAN SEA SFO SJC SMF SNA PSP GEG RNO",
    "America/Anchorage": "ANC", "Pacific/Honolulu": "HNL OGG KOA LIH", "America/Puerto_Rico": "SJU",
}
# CSV uses the more precise IANA zone; identical UTC offset to the old coarse name
EQUIVALENT = {("DTW", "America/New_York"): "America/Detroit", ("IND", "America/New_York"): "America/Indiana/Indianapolis",
              ("BOI", "America/Denver"): "America/Boise"}


def rows():
    with (ROOT / "models" / "airports_ref.csv").open() as f:
        return list(csv.DictReader(f))


def test_csv_columns_and_app_airports_covered():
    r = rows()
    assert list(r[0]) == ["iata", "name", "lat", "lon", "tz", "state", "daily_flights_2025"]
    codes = {x["iata"] for x in r}
    app_ts = ROOT.parent / "flyrely-app" / "src" / "lib" / "airports.ts"
    app = set(re.findall(r"code:\s*'([A-Z]{3})'", app_ts.read_text())) if app_ts.exists() else set(main.AIRPORT_DELAY_RATES)
    assert len(app) >= 30 and app <= codes
    assert all(x["tz"] and x["lat"] and x["lon"] and x["name"] for x in r)
    for x in r:
        ZoneInfo(x["tz"])  # valid IANA names


def test_main_tables_built_from_csv():
    assert set(main.AIRPORT_COORDS) == {x["iata"] for x in rows()}
    assert main.AIRPORT_COORDS["JFK"] == (40.6413, -73.7781)  # legacy coordinates preserved


def test_tz_agrees_with_old_h2h_table():
    for tz, codes in OLD_TZ.items():
        for c in codes.split():
            new = h2h.AIRPORT_TZ[c]
            assert new == tz or EQUIVALENT.get((c, tz)) == new, (c, tz, new)
            now = datetime(2026, 7, 1, 12), datetime(2026, 1, 1, 12)
            assert all(d.replace(tzinfo=ZoneInfo(new)).utcoffset() == d.replace(tzinfo=ZoneInfo(tz)).utcoffset() for d in now)


def test_previously_uncovered_airport_predicts_without_warning():
    assert "FWA" in main.AIRPORT_COORDS
    r = client.post("/predict", json=REQ)
    assert r.status_code == 200
    assert not any("FWA" in w for w in r.json()["warnings"])
    assert main.build_features("FWA", "ATL", datetime(2026, 11, 15, 18), "DL")["distance"] != 250


def test_unknown_airport_is_422():
    r = client.post("/predict", json={**REQ, "origin": "ZZZ"})
    assert r.status_code == 422
    d = r.json()["detail"]
    assert d["reason"] == "unknown_airport" and "ZZZ" in d["message"]
    assert client.post("/predict", json={**REQ, "destination": "QQQ"}).status_code == 422


def test_same_origin_and_destination_is_422():
    r = client.post("/predict", json={**REQ, "origin": "JFK", "destination": "jfk"})
    assert r.status_code == 422
    assert r.json()["detail"]["reason"] == "same_airport"


def test_every_scheduled_airport_is_in_the_reference():
    import gzip
    with gzip.open(ROOT / "models" / "route_schedule.csv.gz", "rt") as f:
        r = csv.DictReader(f)
        seen = set()
        for row in r:
            seen.add(row["origin"]); seen.add(row["dest"])
    assert seen - set(main.AIRPORT_COORDS) == set()
