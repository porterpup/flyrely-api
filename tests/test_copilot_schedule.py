"""copilot/schedule.py and scripts/build_route_schedule.py, on a small fixture (not the real artifact)."""

import gzip
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from copilot import schedule as sched  # noqa: E402
import build_route_schedule as brs  # noqa: E402

HEADER = ("marketing_airline,marketing_flight_number,origin,dest,dep_time,arr_time,operating_airline,"
          "operating_flight_number,dow_mask,days_seen,schedule_as_of\n")
ROWS = [
    "AA,117,JFK,LAX,16:00,19:25,AA,117,127,8,2026-08",     # every day
    "DL,50,JFK,LAX,07:15,10:40,DL,50,31,6,2026-08",        # Mon-Fri (bits 0-4)
    "UA,900,JFK,LAX,12:00,15:30,OO,5521,96,4,2026-08",     # Sat+Sun only (bits 5,6)
]


@pytest.fixture
def csv_gz(tmp_path):
    p = tmp_path / "route_schedule.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write(HEADER + "\n".join(ROWS) + "\n")
    return p


def test_weekday_filter_and_order(csv_gz):
    s = sched.BtsTypicalSchedule(csv_gz)
    mon = s.flights_on("JFK", "LAX", date(2026, 11, 16))        # a Monday
    assert [f.flight_number for f in mon] == ["DL50", "AA117"]  # earliest departure first
    sat = s.flights_on("JFK", "LAX", "2026-11-14")              # a Saturday
    assert [f.flight_number for f in sat] == ["UA900", "AA117"]
    assert sat[0].operating_airline == "OO" and sat[0].operating_flight_number == "OO5521"
    assert s.schedule_as_of() == "2026-08"
    assert s.flights_on("LAX", "JFK", date(2026, 11, 16)) == []


def test_lazy_load(csv_gz):
    s = sched.BtsTypicalSchedule(csv_gz)
    assert s._by_route is None
    s.flights_on("JFK", "LAX", date(2026, 11, 16))
    assert s._by_route is not None


def test_missing_file_is_empty(tmp_path):
    s = sched.BtsTypicalSchedule(tmp_path / "nope.csv.gz")
    assert s.flights_on("JFK", "LAX", date(2026, 11, 16)) == [] and s.schedule_as_of() is None


def test_swim_stub_not_built():
    with pytest.raises(NotImplementedError):
        sched.SwimScheduleSource().flights_on("JFK", "LAX", date(2026, 11, 16))


def test_build_from_synthetic_months():
    days = pd.date_range("2026-08-03", periods=14)            # two weeks, Mon..Sun twice
    rows = []
    for d in days:
        rows.append(dict(FlightDate=d, Marketing_Airline_Network="UA", Flight_Number_Marketing_Airline=900,
                         Reporting_Airline="OO", Flight_Number_Reporting_Airline=5521, Origin="JFK", Dest="LAX",
                         CRSDepTime=1200, CRSArrTime=1530, Cancelled=0.0))
    rows.append(dict(rows[0], FlightDate=pd.Timestamp("2026-08-04"), Flight_Number_Marketing_Airline=7,
                     Cancelled=1.0))                                     # cancelled: not counted
    rows.append(dict(rows[0], Flight_Number_Marketing_Airline=8))        # seen once: dropped
    out = brs.build(pd.DataFrame(rows), "2026-08")
    assert len(out) == 1
    r = out.iloc[0]
    assert (r.dep_time, r.arr_time, r.operating_airline, r.dow_mask, r.days_seen) == ("12:00", "15:30", "OO", 127, 14)


def test_hhmm():
    assert brs.hhmm(pd.Series([5, 1200, 2400])).tolist() == ["00:05", "12:00", "00:00"]


def test_only_months_before_holdout():
    assert all(m < "2026-09" for m in brs.allowed_months())
