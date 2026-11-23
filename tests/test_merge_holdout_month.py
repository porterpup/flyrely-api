"""backtest/merge_holdout_month.py: appends only the month's rows, is idempotent, backs up, refuses a locked month.
Offline: everything lives in tmp_path (no real data/)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))

import holdout  # noqa: E402
import merge_holdout_month as mh  # noqa: E402

MONTH = "2026-09"
HDR = "station,valid,vsby,metar\n"
ZIP = "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_2026_9.zip"
PCOLS = ["adv_date", "advn", "element", "center", "title", "kind", "status", "sent_utc", "list_date", "period_start",
         "period_end"]


def prog(list_dates):
    n = len(list_dates)
    return pd.DataFrame({"adv_date": ["x"] * n, "advn": range(n), "element": ["ATL"] * n, "center": ["ZTL"] * n,
                         "title": ["t"] * n, "kind": ["GS"] * n, "status": ["active"] * n,
                         "sent_utc": pd.to_datetime(["2026-09-01"] * n), "list_date": list_dates,
                         "period_start": pd.to_datetime([None] * n), "period_end": pd.to_datetime([None] * n)})


def wx(hours, code_dtype="float64"):
    return pd.DataFrame({"airport": "ATL", "utc_hour": pd.to_datetime(hours), "weather_code_previous_day7": 1}).astype(
        {"weather_code_previous_day7": code_dtype})


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    h = d / "holdout" / MONTH
    for sub in ("bts", "metar", "faa_adv/list", "weather"):
        (h / sub).mkdir(parents=True)
        if sub != "faa_adv/list":
            (d / sub.split("/")[0]).mkdir(parents=True, exist_ok=True)
    (d / "bts" / "parquet_v2").mkdir()
    (d / "bts" / "marketing").mkdir()
    (d / "faa_adv" / "list").mkdir(parents=True)
    # BTS
    (h / "bts" / ZIP).write_bytes(b"zip")
    pd.DataFrame({"FlightDate": pd.to_datetime(["2026-09-01", "2026-09-30"])}).to_parquet(h / "bts" / f"{MONTH}.parquet")
    # METAR: main holds August, the holdout file holds September plus the first day of October
    (d / "metar" / "ATL_2026.csv").write_text(HDR + "ATL,2026-08-31 23:53,10,aug\n")
    (h / "metar" / f"ATL_{MONTH}.csv").write_text(
        HDR + "ATL,2026-09-01 00:53,10,sep1\nATL,2026-09-30 23:53,10,sep30\nATL,2026-10-01 00:53,10,oct1\n")
    # FAA
    for day in ("2026-09-01", "2026-09-30", "2026-10-01"):
        (h / "faa_adv" / "list" / f"{day}.html").write_text(day)
    prog(["2026-08-31"]).to_parquet(d / "faa_adv" / "programs_v2.parquet")
    prog(["2026-09-01", "2026-09-30", "2026-10-01"]).to_parquet(h / "faa_adv" / "programs.parquet")
    # weather (holdout code column int64, main float64)
    wx(["2026-08-31 23:00"]).to_parquet(d / "weather" / "ATL.parquet")
    wx(["2026-09-01 00:00", "2026-09-30 23:00", "2026-10-01 00:00"], "int64").to_parquet(h / "weather" / "ATL.parquet")
    return d


def tree(d: Path):
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def test_merge_appends_only_the_month_and_backs_up(data):
    held = tree(data / "holdout")
    mh.Merger(MONTH, data).run()
    lines = (data / "metar" / "ATL_2026.csv").read_text().splitlines()
    assert lines == [HDR.strip(), "ATL,2026-08-31 23:53,10,aug", "ATL,2026-09-01 00:53,10,sep1",
                     "ATL,2026-09-30 23:53,10,sep30"]                              # October 1 stays out
    assert (data / "bts" / "marketing" / ZIP).read_bytes() == b"zip"
    assert (data / "bts" / "parquet_v2" / f"{MONTH}.parquet").exists()
    assert sorted(p.name for p in (data / "faa_adv" / "list").iterdir()) == ["2026-09-01.html", "2026-09-30.html"]
    p = pd.read_parquet(data / "faa_adv" / "programs_v2.parquet")
    assert list(p["list_date"]) == ["2026-08-31", "2026-09-01", "2026-09-30"]
    w = pd.read_parquet(data / "weather" / "ATL.parquet")
    assert list(w["utc_hour"].astype(str)) == ["2026-08-31 23:00:00", "2026-09-01 00:00:00", "2026-09-30 23:00:00"]
    assert str(w["weather_code_previous_day7"].dtype) == "float64"
    b = data / "backup" / f"holdout_merge_{MONTH}"
    assert (b / "metar" / "ATL_2026.csv").read_text() == HDR + "ATL,2026-08-31 23:53,10,aug\n"
    assert len(pd.read_parquet(b / "faa_adv" / "programs_v2.parquet")) == 1
    assert len(pd.read_parquet(b / "weather" / "ATL.parquet")) == 1
    assert tree(data / "holdout") == held                                          # the holdout folder is untouched


def test_merge_is_idempotent_and_never_overwrites_a_backup(data):
    mh.Merger(MONTH, data).run()
    after = tree(data)
    mh.Merger(MONTH, data).run()
    assert tree(data) == after


def test_dry_run_changes_nothing(data):
    before = tree(data)
    mh.Merger(MONTH, data, dry_run=True).run()
    assert tree(data) == before


def test_locked_month_is_refused(data):
    for m in ("2026-10", "2026-11", "2027-01"):
        with pytest.raises(SystemExit, match="protected"):
            mh.Merger(m, data)
    assert holdout.is_locked_month("2026-10")


def test_a_different_existing_file_stops_the_merge(data):
    (data / "bts" / "marketing" / ZIP).write_bytes(b"other")
    with pytest.raises(SystemExit, match="differs"):
        mh.Merger(MONTH, data).run()
    assert (data / "bts" / "marketing" / ZIP).read_bytes() == b"other"


def test_bts_rows_outside_the_month_are_refused(data):
    pd.DataFrame({"FlightDate": pd.to_datetime(["2026-09-30", "2026-10-01"])}).to_parquet(
        data / "holdout" / MONTH / "bts" / f"{MONTH}.parquet")
    with pytest.raises(RuntimeError, match="outside"):
        mh.Merger(MONTH, data).run()
