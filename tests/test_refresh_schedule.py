"""scripts/refresh_schedule.py: holdout gate, sanity checks, atomic swap; copilot/schedule.py mtime reload.
No network: urllib is patched to fail loudly in every test."""

import gzip
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import refresh_schedule as rs  # noqa: E402
from copilot import schedule as sched  # noqa: E402

HEADER = ("marketing_airline,marketing_flight_number,origin,dest,dep_time,arr_time,operating_airline,"
          "operating_flight_number,dow_mask,days_seen,schedule_as_of\n")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used in a test")
    monkeypatch.setattr(urllib.request, "urlopen", boom)


def schedule_df(n_per=10, as_of="2026-08", carriers=("AA", "DL", "UA", "WN", "AS", "B6"), hawaii=25):
    rows = []
    for c in carriers:
        for i in range(n_per):
            rows.append((c, str(100 + i), "JFK", "LAX", "08:00", "11:00", c, str(100 + i), 127, 8, as_of))
    for i in range(hawaii):
        rows.append(("AS", str(900 + i), "SEA", "HNL", "08:00", "11:00", "AS", str(900 + i), 127, 8, as_of))
    return pd.DataFrame(rows, columns=HEADER.strip().split(","))


def write_gz(path, df):
    with gzip.open(path, "wt") as f:
        f.write(df.to_csv(index=False))


# ---------------------------------------------------------------- gate
def test_gate_closed_is_a_noop_with_no_network(tmp_path, caplog):
    caplog.set_level("INFO")
    sch = tmp_path / "route_schedule.csv.gz"
    write_gz(sch, schedule_df())
    before = sch.read_bytes()
    out = rs.refresh(schedule=sch, extra_dir=tmp_path / "extra", main_dir=tmp_path / "main",
                     result=tmp_path / "h1_b5_1_2026-09.json")    # score result absent
    assert out == "gate-closed"
    assert "holdout not yet scored, skipping" in caplog.text
    assert sch.read_bytes() == before
    assert not (tmp_path / "extra").exists()            # nothing created, nothing read
    assert not (tmp_path / "route_schedule.csv.gz.prev").exists()


def test_gate_opens_only_with_score_result(tmp_path):
    f = tmp_path / "h1_b5_1_2026-09.json"
    assert not rs.gate_open(f)
    f.write_text("{}")
    assert rs.gate_open(f)


def test_gate_default_path_is_the_score_output():
    assert rs.SCORE_RESULT == ROOT / "backtest" / "results" / "h1_b5_1_2026-09.json"


def test_main_closed_gate_never_touches_network_or_unlock(monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "SCORE_RESULT", tmp_path / "missing.json")
    monkeypatch.setenv("FLYRELY_HOLDOUT_UNLOCK", "2026-09")
    monkeypatch.setenv("FLYRELY_ENV_FILE", str(tmp_path / "none.env"))
    assert rs.main([]) == 0
    assert "FLYRELY_HOLDOUT_UNLOCK" not in os.environ


def test_open_gate_but_nothing_new(tmp_path, monkeypatch):
    res = tmp_path / "r.json"
    res.write_text("{}")
    sch = tmp_path / "route_schedule.csv.gz"
    write_gz(sch, schedule_df(as_of="2026-09"))
    main = tmp_path / "main"
    main.mkdir()
    (main / "2026-09.parquet").write_bytes(b"x")          # newest on disk == schedule_as_of; never read
    monkeypatch.setattr(rs, "month_published", lambda ym: False)
    assert rs.refresh(schedule=sch, extra_dir=tmp_path / "extra", main_dir=main, result=res) == "up-to-date"


# ---------------------------------------------------------------- sanity
def test_sanity_ok():
    s = rs.sanity_check(schedule_df(), old_rows=90, old_as_of="2026-08")
    assert s["rows"] == 85 and s["AS_hawaii_rows"] == 25


@pytest.mark.parametrize("kwargs,old_rows,msg", [
    (dict(n_per=2), 85, "row count"),                                  # far fewer rows
    (dict(n_per=30), 85, "row count"),                                 # far more rows
    (dict(carriers=("AA", "DL", "UA", "WN", "AS")), None, "major carriers missing"),
    (dict(hawaii=3), None, "Hawaii"),
])
def test_sanity_failures(kwargs, old_rows, msg):
    with pytest.raises(rs.SanityError, match=msg):
        rs.sanity_check(schedule_df(**kwargs), old_rows=old_rows)


def test_sanity_rejects_nk_and_older_as_of():
    df = schedule_df()
    df.loc[0, "marketing_airline"] = "NK"
    with pytest.raises(rs.SanityError):
        rs.sanity_check(df)
    with pytest.raises(rs.SanityError, match="older"):
        rs.sanity_check(schedule_df(as_of="2026-07"), old_as_of="2026-08")


def test_build_drops_nk():
    import build_route_schedule as brs
    d = pd.DataFrame({"FlightDate": pd.to_datetime(["2026-08-03", "2026-08-10"] * 2),
                      "Marketing_Airline_Network": ["NK", "NK", "AA", "AA"],
                      "Flight_Number_Marketing_Airline": [1, 1, 2, 2], "Reporting_Airline": ["NK", "NK", "AA", "AA"],
                      "Flight_Number_Reporting_Airline": [1, 1, 2, 2], "Origin": "JFK", "Dest": "LAX",
                      "CRSDepTime": 800, "CRSArrTime": 1100, "Cancelled": 0.0})
    out = brs.build(d, "2026-08")
    assert set(out["marketing_airline"]) == {"AA"}


# ---------------------------------------------------------------- swap
def test_swap_keeps_prev_and_is_atomic(tmp_path):
    cur, new = tmp_path / "route_schedule.csv.gz", tmp_path / "route_schedule.csv.gz.new"
    cur.write_bytes(b"old")
    new.write_bytes(b"new")
    rs.swap_in(new, cur)
    assert cur.read_bytes() == b"new"
    assert (tmp_path / "route_schedule.csv.gz.prev").read_bytes() == b"old"
    assert not new.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["route_schedule.csv.gz", "route_schedule.csv.gz.prev"]


def test_swap_failure_leaves_current_untouched(tmp_path, monkeypatch):
    cur, new = tmp_path / "route_schedule.csv.gz", tmp_path / "route_schedule.csv.gz.new"
    cur.write_bytes(b"old")
    new.write_bytes(b"new")
    real = os.replace

    def flaky(a, b):
        if str(b) == str(cur):
            raise OSError("disk")
        return real(a, b)
    monkeypatch.setattr(rs.os, "replace", flaky)
    with pytest.raises(OSError):
        rs.swap_in(new, cur)
    assert cur.read_bytes() == b"old"


def test_failed_sanity_leaves_schedule_and_cleans_tmp(tmp_path, monkeypatch):
    res = tmp_path / "r.json"
    res.write_text("{}")
    sch = tmp_path / "route_schedule.csv.gz"
    write_gz(sch, schedule_df(as_of="2026-08"))
    before = sch.read_bytes()
    main = tmp_path / "main"
    main.mkdir()
    (main / "2026-08.parquet").write_bytes(b"x")
    monkeypatch.setattr(rs, "fetch_new_months", lambda e, m: ["2026-09"])
    monkeypatch.setattr(rs, "build_new", lambda tmp, e, m: (tmp.write_bytes(b"junk"), schedule_df(n_per=2))[1])
    with pytest.raises(rs.SanityError):
        rs.refresh(schedule=sch, extra_dir=tmp_path / "extra", main_dir=main, result=res)
    assert sch.read_bytes() == before
    assert not (tmp_path / "route_schedule.csv.gz.new").exists()
    assert not (tmp_path / "route_schedule.csv.gz.prev").exists()


def test_full_refresh_with_fake_month(tmp_path, monkeypatch):
    res = tmp_path / "r.json"
    res.write_text("{}")
    sch = tmp_path / "route_schedule.csv.gz"
    write_gz(sch, schedule_df(as_of="2026-08"))
    main, extra = tmp_path / "main", tmp_path / "extra"
    main.mkdir()
    extra.mkdir()
    (main / "2026-08.parquet").write_bytes(b"x")
    new_df = schedule_df(as_of="2026-09")
    monkeypatch.setattr(rs, "fetch_new_months", lambda e, m: ["2026-09"])
    monkeypatch.setattr(rs, "local_months", lambda m=None, e=None: ["2026-08", "2026-09"])
    monkeypatch.setattr(rs, "build_new", lambda tmp, e, m: (rs.brs._write(new_df, tmp), new_df)[1])
    assert rs.refresh(schedule=sch, extra_dir=extra, main_dir=main, result=res) == "refreshed 2026-09"
    assert rs.schedule_as_of(sch) == "2026-09"
    assert rs.schedule_as_of(tmp_path / "route_schedule.csv.gz.prev") == "2026-08"


def test_next_month():
    assert rs.next_month("2026-09") == "2026-10" and rs.next_month("2026-12") == "2027-01"


# ---------------------------------------------------------------- mtime reload
def test_loader_reloads_when_file_changes(tmp_path):
    p = tmp_path / "route_schedule.csv.gz"
    write_gz(p, schedule_df(as_of="2026-08"))
    s = sched.BtsTypicalSchedule(p)
    assert s.schedule_as_of() == "2026-08"
    assert len(s.flights_on("JFK", "LAX", "2026-11-16")) == 60
    new = tmp_path / "n.gz"
    write_gz(new, schedule_df(n_per=1, as_of="2026-09"))
    os.replace(new, p)                                   # what refresh_schedule.swap_in does
    assert s.schedule_as_of() == "2026-09"
    assert len(s.flights_on("JFK", "LAX", "2026-11-16")) == 6


def test_loader_does_not_reparse_unchanged_file(tmp_path, monkeypatch):
    p = tmp_path / "route_schedule.csv.gz"
    write_gz(p, schedule_df())
    s = sched.BtsTypicalSchedule(p)
    s.flights_on("JFK", "LAX", "2026-11-16")
    table = s._by_route
    s.flights_on("JFK", "LAX", "2026-11-16")
    assert s._by_route is table


def test_loader_keeps_last_good_table_if_file_goes_bad_or_missing(tmp_path):
    p = tmp_path / "route_schedule.csv.gz"
    write_gz(p, schedule_df(as_of="2026-08"))
    s = sched.BtsTypicalSchedule(p)
    assert s.schedule_as_of() == "2026-08"
    p.write_bytes(b"not gzip")
    assert s.schedule_as_of() == "2026-08"
    p.unlink()
    assert s.schedule_as_of() == "2026-08"
    assert s.flights_on("JFK", "LAX", "2026-11-16")


def test_flights_by_number_index_follows_reload(tmp_path):
    p = tmp_path / "route_schedule.csv.gz"
    write_gz(p, schedule_df(as_of="2026-08"))
    s = sched.BtsTypicalSchedule(p)
    assert s.flights_by_number("AA", "100", "2026-11-16")
    new = tmp_path / "n.gz"
    write_gz(new, schedule_df(n_per=1, as_of="2026-09").assign(marketing_flight_number="777"))
    os.replace(new, p)
    assert s.flights_by_number("AA", "100", "2026-11-16") == []
    assert s.flights_by_number("AA", "777", "2026-11-16")
