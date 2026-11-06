"""H1 holdout guards (PROTOCOL_B5_1_HOLDOUT.md section 8). Offline only: no network, no real holdout data, no
real data/ folder (CI has none). Every fetch is faked and every write lands in tmp_path."""

import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backtest"))

import backtest as bt  # noqa: E402
import experiments_b5 as e  # noqa: E402
import fetch_faa_adv as ff  # noqa: E402
import fetch_metar  # noqa: E402
import fetch_weather  # noqa: E402
import holdout  # noqa: E402
import prepare  # noqa: E402
from flyrely_features import metar, weather  # noqa: E402

MONTH = "2026-09"
UNLOCK = holdout.UNLOCK_VAR


@pytest.fixture(autouse=True)
def locked(monkeypatch):
    monkeypatch.delenv(UNLOCK, raising=False)


@pytest.fixture
def hroot(tmp_path, monkeypatch):
    """Holdout root and frozen airport list redirected into tmp_path."""
    monkeypatch.setattr(holdout, "HOLDOUT_ROOT", tmp_path / "holdout")
    monkeypatch.setattr(holdout, "MODELS_DIR", tmp_path / "holdout" / "models")
    af = tmp_path / "h1_airports.json"
    af.write_text(json.dumps({"metar": ["ATL", "ORD"], "weather": ["ATL", "ORD", "SEA"], "faa_covered": ["ATL"]}))
    monkeypatch.setattr(holdout, "AIRPORTS_FILE", af)
    return tmp_path / "holdout"


def under(path: Path, root: Path) -> bool:
    return root in path.parents or path == root


# ---------------------------------------------------------------- the policy

def test_unlock_opens_only_the_named_month(monkeypatch):
    assert holdout.unlocked_month(MONTH, {}) is None                                   # nothing set
    assert holdout.unlocked_month(None, {UNLOCK: MONTH}) is None                       # env without the flag
    assert holdout.unlocked_month(MONTH, {UNLOCK: MONTH}) == MONTH                     # both, same month
    assert holdout.unlocked_month(MONTH, {UNLOCK: "2026-10"}) is None                  # another month's unlock
    assert holdout.unlocked_month("2026-10", {UNLOCK: MONTH}) is None                  # another month's flag
    assert holdout.unlocked_month("2026-05", {UNLOCK: "2026-05"}) is None              # not a protected month
    monkeypatch.setenv(UNLOCK, MONTH)
    assert holdout.require_unlock(MONTH) == MONTH
    for other in ("2026-10", "2026-11", "2027-01"):
        with pytest.raises(SystemExit):
            holdout.require_unlock(other)


def test_only_the_allowlisted_month_can_be_unlocked(monkeypatch):
    assert holdout.UNLOCKABLE == ("2026-09",)
    for m in ("2026-10", "2026-11", "2027-01"):
        assert holdout.unlocked_month(m, {UNLOCK: m}) is None          # named by both, still refused
        monkeypatch.setenv(UNLOCK, m)
        with pytest.raises(SystemExit, match="only"):
            holdout.require_unlock(m)
        with pytest.raises(SystemExit):
            holdout.check_end_date(f"{m}-02", m)


def test_old_unlock_value_one_is_refused(monkeypatch):
    with pytest.raises(SystemExit) as err:
        holdout.unlocked_month(MONTH, {UNLOCK: "1"})
    assert "YYYY-MM" in str(err.value) or MONTH in str(err.value)
    monkeypatch.setenv(UNLOCK, "1")
    with pytest.raises(SystemExit):
        holdout.require_unlock(MONTH)
    with pytest.raises(SystemExit):
        holdout.check_end_date("2026-09-30")                                           # even without the flag


def test_garbage_months_are_rejected(monkeypatch):
    for bad in ("2026-9", "2026-13", "garbage", "", "2026-09-01"):
        with pytest.raises(SystemExit):
            holdout.require_unlock(bad)
    with pytest.raises(SystemExit):
        holdout.unlocked_month(MONTH, {UNLOCK: "nine"})


def test_locked_range_starts_at_september_2026():
    assert holdout.LOCKED_FROM == "2026-09"
    assert str(holdout.LAST_ALLOWED_DATE) == "2026-08-31"
    assert holdout.is_locked_month("2026-09") and holdout.is_locked_month("2027-01")
    assert not holdout.is_locked_month("2026-08")
    assert holdout.is_locked_date("2026-09-01") and not holdout.is_locked_date("2026-08-31")
    assert [str(d) for d in holdout.input_window("2026-09")] == ["2026-09-01", "2026-10-01"]


def test_sh_mirror_uses_the_same_locked_start():
    text = (ROOT / "backtest" / "fetch_bts.sh").read_text()
    assert f"HOLDOUT_START={holdout.LOCKED_FROM}" in text


# ---------------------------------------------------------------- every fetcher refuses without the flag

def test_fetchers_refuse_september_without_the_flag(monkeypatch):
    for env in (None, MONTH, "1"):                    # even an unlock that names the month is useless without --holdout
        if env:
            monkeypatch.setenv(UNLOCK, env)
        for mod in (fetch_metar, fetch_weather, ff):
            for end in ("2026-09-01", "2026-09-30", "2026-10-01", "2027-01-01"):
                with pytest.raises(SystemExit):
                    mod.check_holdout(end)
            mod.check_holdout("2026-08-31")
        with pytest.raises(SystemExit):
            fetch_metar.main(["--end", "2026-09-30"])
        with pytest.raises(SystemExit):
            fetch_weather.main(["--end", "2026-09-30"])
        monkeypatch.setattr(sys, "argv", ["x", "lists", "--end", "2026-09-30"])
        with pytest.raises(SystemExit):
            ff.main()
    monkeypatch.delenv(UNLOCK)


def test_holdout_flag_needs_the_matching_unlock(hroot, monkeypatch):
    for argv_main in (lambda: fetch_metar.main(["--holdout", MONTH]), lambda: fetch_weather.main(["--holdout", MONTH])):
        with pytest.raises(SystemExit):                                   # locked
            argv_main()
        monkeypatch.setenv(UNLOCK, "2026-10")
        with pytest.raises(SystemExit):                                   # another month's unlock
            argv_main()
        monkeypatch.delenv(UNLOCK)
    for cmd in ("lists", "build"):
        monkeypatch.setattr(sys, "argv", ["x", cmd, "--holdout", MONTH])
        with pytest.raises(SystemExit):
            ff.main()
    assert not hroot.exists()


def test_holdout_refuses_october_and_november_dates(monkeypatch):
    monkeypatch.setenv(UNLOCK, MONTH)
    holdout.check_end_date("2026-10-01", MONTH)                           # month plus one day is the window
    for bad in ("2026-10-02", "2026-11-01", "2026-08-31"):
        with pytest.raises(SystemExit):
            holdout.check_end_date(bad, MONTH)
    with pytest.raises(SystemExit):
        fetch_metar.check_holdout("2026-11-05", MONTH)
    with pytest.raises(SystemExit):                                       # fetch_one re-checks every request
        fetch_metar.fetch_one(None, "ATL", pd.Timestamp("2026-10-01").date(), pd.Timestamp("2026-10-02").date(),
                              holdout_month=MONTH)
    for argv in (["--holdout", MONTH, "--end", "2026-11-30"], ["--holdout", MONTH, "--airports", "ATL"]):
        with pytest.raises(SystemExit):
            fetch_metar.main(argv)
        with pytest.raises(SystemExit):
            fetch_weather.main(argv)


# ---------------------------------------------------------------- fakes for the fetchers

class FakeResp:
    status_code = 200
    text = "station,valid,vsby\nATL,2026-09-01 00:53,10\n"

    def raise_for_status(self):
        pass


class FakeClient:
    calls = []

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None, **k):
        FakeClient.calls.append((url, params))
        return FakeResp()


def snapshot(root: Path) -> set:
    return {str(p.relative_to(root)) for p in root.rglob("*")} if root.exists() else set()


def test_metar_holdout_writes_only_the_holdout_folder(hroot, tmp_path, monkeypatch):
    main_dir = tmp_path / "main_metar"
    monkeypatch.setattr(fetch_metar, "OUT", main_dir)
    monkeypatch.setattr(fetch_metar.httpx, "Client", FakeClient)
    monkeypatch.setattr(fetch_metar.time, "sleep", lambda s: None)
    monkeypatch.setattr(fetch_metar, "load_table", lambda: None)
    monkeypatch.setattr(fetch_metar, "iem_station", lambda s, t: s)
    monkeypatch.setenv(UNLOCK, MONTH)
    FakeClient.calls = []
    fetch_metar.main(["--holdout", MONTH])
    out = hroot / MONTH / "metar"
    assert sorted(p.name for p in out.iterdir()) == ["ATL_2026-09.csv", "ORD_2026-09.csv"]     # the frozen list
    assert not main_dir.exists()
    assert len(FakeClient.calls) == 2
    p = FakeClient.calls[0][1]
    assert (p["year1"], p["month1"], p["day1"]) == (2026, 9, 1) and (p["month2"], p["day2"]) == (10, 2)


def test_weather_holdout_writes_only_the_holdout_folder(hroot, tmp_path, monkeypatch):
    main_dir = tmp_path / "main_weather"
    ref = tmp_path / "ref.csv"
    pd.DataFrame({"iata": ["ATL", "ORD", "SEA"], "lat": [1.0, 2.0, 3.0], "lon": [1.0, 2.0, 3.0]}).to_csv(ref, index=False)
    seen = []

    def fake_fetch(client, iata, lat, lon, end, start):
        seen.append((iata, start, end))
        return pd.DataFrame({"airport": [iata], "utc_hour": [pd.Timestamp(start)]})

    monkeypatch.setattr(fetch_weather, "OUT", main_dir)
    monkeypatch.setattr(fetch_weather, "REF", ref)
    monkeypatch.setattr(fetch_weather, "fetch", fake_fetch)
    monkeypatch.setattr(fetch_weather.httpx, "Client", FakeClient)
    monkeypatch.setattr(fetch_weather.time, "sleep", lambda s: None)
    monkeypatch.setenv(UNLOCK, MONTH)
    fetch_weather.main(["--holdout", MONTH])
    assert sorted(p.name for p in (hroot / MONTH / "weather").iterdir()) == ["ATL.parquet", "ORD.parquet", "SEA.parquet"]
    assert not main_dir.exists()
    assert {(s, en) for _, s, en in seen} == {("2026-09-01", "2026-10-01")}


def test_faa_holdout_lists_and_build_stay_in_the_holdout_folder(hroot, tmp_path, monkeypatch):
    main_dir = tmp_path / "main_faa"
    page = ('<table><tr><td><a href="/adv/adv_otherdis?adv_date=09152026&advn=12">012 </a></td><td>ATL/ZTL</td>'
            '<td>09/15/26</td><td>CDM GROUND STOP</td><td>09/15/26 22:12</td></tr></table>' + " " * 600)
    monkeypatch.setattr(ff, "OUT", main_dir)
    monkeypatch.setattr(ff, "get", lambda client, url, params=None, tries=6: page)
    monkeypatch.setattr(ff.httpx, "Client", FakeClient)
    monkeypatch.setattr(ff.time, "sleep", lambda s: None)
    monkeypatch.setenv(UNLOCK, MONTH)
    monkeypatch.setattr(sys, "argv", ["x", "lists", "--holdout", MONTH])
    ff.main()
    pages = sorted(p.name for p in (hroot / MONTH / "faa_adv" / "list").iterdir())
    assert pages[0] == "2026-09-01.html" and pages[-1] == "2026-10-01.html" and len(pages) == 31
    monkeypatch.setattr(sys, "argv", ["x", "build", "--holdout", MONTH])
    ff.main()
    prog = pd.read_parquet(hroot / MONTH / "faa_adv" / "programs.parquet")
    assert len(prog) == 31 and set(prog["kind"]) == {"GS"}
    assert not main_dir.exists()                                          # nothing reached the main folder
    monkeypatch.setattr(sys, "argv", ["x", "details", "--holdout", MONTH])
    with pytest.raises(SystemExit) as err:
        ff.main()
    assert "details" in str(err.value)


def test_list_frame_never_merges_holdout_pages(hroot, tmp_path, monkeypatch):
    page = ('<table><tr><td><a href="/adv/adv_otherdis?adv_date=09152026&advn=12">012 </a></td><td>ATL/ZTL</td>'
            '<td>09/15/26</td><td>CDM GROUND STOP</td><td>09/15/26 22:12</td></tr></table>')
    main_dir = tmp_path / "main_faa"
    (main_dir / "list").mkdir(parents=True)
    (main_dir / "list" / "2026-08-31.html").write_text(page)
    (main_dir / "list" / "2026-09-15.html").write_text(page)             # a stray holdout page in the main folder
    monkeypatch.setattr(ff, "OUT", main_dir)
    assert list(ff.list_frame()["list_date"]) == ["2026-08-31"]
    hl = hroot / MONTH / "faa_adv" / "list"
    hl.mkdir(parents=True)
    for d in ("2026-09-15", "2026-10-01", "2026-10-02"):
        (hl / f"{d}.html").write_text(page)
    assert list(ff.list_frame(MONTH)["list_date"]) == ["2026-09-15", "2026-10-01"]   # only the month plus one day


def test_busiest_airports_ignore_holdout_files_in_both_fetchers(tmp_path, monkeypatch):
    for ym, orig in (("2025-01", "ATL"), ("2026-08", "ATL"), ("2026-09", "ZZZ"), ("2026-10", "ZZZ"), ("2027-01", "ZZZ")):
        pd.DataFrame({"Origin": [orig] * 5, "Dest": ["ORD"] * 5}).to_parquet(tmp_path / f"{ym}.parquet")
    for mod in (fetch_metar, fetch_weather):
        monkeypatch.setattr(mod, "PARQUET", tmp_path)
        assert set(mod.busiest_airports(10)) == {"ATL", "ORD"}


# ---------------------------------------------------------------- BTS: prepare.py, load_months, fetch_bts.sh

def bts_zip(path: Path, month: str, extra_dates=()):
    y, m = int(month[:4]), int(month[5:])
    header = ('"FlightDate","Marketing_Airline_Network","Flight_Number_Marketing_Airline","Operating_Airline ",'
              '"Tail_Number","Flight_Number_Operating_Airline","Origin","Dest","CRSDepTime","DepDelay",'
              '"TaxiOut","WheelsOff","CRSArrTime","Cancelled","CancellationCode","Diverted","CRSElapsedTime","Distance",')
    rows = [f"{d},DL,1,9E,N1,1,ATL,ORD,0630,5.0,10.0,0640,0830,0.0,,0.0,95.0,508.0," for d in (f"{month}-03", *extra_dates)]
    zp = path / f"On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{y}_{m}.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("data.csv", "\n".join([header, *rows]) + "\n")
    return zp


def test_prepare_holdout_converts_only_that_month_into_its_folder(hroot, monkeypatch):
    d = hroot / MONTH / "bts"
    d.mkdir(parents=True)
    bts_zip(d, MONTH)
    bts_zip(d, "2026-10")                                               # a stray October zip is never converted
    with pytest.raises(SystemExit):                                     # locked
        prepare.main(["--source", "marketing", "--holdout", MONTH])
    monkeypatch.setenv(UNLOCK, MONTH)
    prepare.main(["--source", "marketing", "--holdout", MONTH])
    assert sorted(p.name for p in d.glob("*.parquet")) == ["2026-09.parquet"]
    with pytest.raises(SystemExit):
        prepare.main(["--holdout", MONTH])                              # reporting source has no holdout form


def test_prepare_holdout_refuses_october_rows(hroot, monkeypatch):
    d = hroot / MONTH / "bts"
    d.mkdir(parents=True)
    bts_zip(d, MONTH, extra_dates=("2026-10-01",))
    monkeypatch.setenv(UNLOCK, MONTH)
    with pytest.raises(RuntimeError, match="outside"):
        prepare.main(["--source", "marketing", "--holdout", MONTH])
    assert not list(d.glob("*.parquet"))


def test_prepare_without_the_flag_never_converts_a_holdout_zip(tmp_path, monkeypatch):
    src = tmp_path / "marketing"
    src.mkdir()
    bts_zip(src, "2026-08")
    bts_zip(src, MONTH)
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setenv(UNLOCK, MONTH)
    prepare.main(["--source", "marketing"])
    assert [p.name for p in (tmp_path / "parquet_v2").glob("*.parquet")] == ["2026-08.parquet"]
    with pytest.raises(SystemExit):
        prepare.convert(next(src.glob("*_2026_9.zip")), tmp_path / "parquet_v2", "marketing")


def month_frame(month: str, dates):
    n = len(dates)
    return pd.DataFrame({"FlightDate": pd.to_datetime(dates), "Reporting_Airline": ["DL"] * n,
                         "Marketing_Airline_Network": ["DL"] * n, "Origin": ["ATL"] * n, "Dest": ["ORD"] * n,
                         "CRSDepTime": [630] * n, "DepDelay": [5.0] * n, "Cancelled": [0.0] * n, "Diverted": [0.0] * n})


def test_load_months_never_reads_a_holdout_month_from_the_main_folder(tmp_path, hroot, monkeypatch):
    main = tmp_path / "parquet_v2"
    main.mkdir()
    month_frame("2026-08", ["2026-08-03"]).to_parquet(main / "2026-08.parquet")
    month_frame(MONTH, ["2026-09-03"]).to_parquet(main / "2026-09.parquet")      # a stray file
    monkeypatch.setattr(bt, "PARQUET", main)
    assert len(bt.load_months("2026-08", "2026-08")) == 1
    with pytest.raises(RuntimeError, match="protected holdout"):
        bt.load_months("2026-08", MONTH)
    hd = hroot / MONTH / "bts"
    hd.mkdir(parents=True)
    month_frame(MONTH, ["2026-09-04"]).to_parquet(hd / "2026-09.parquet")
    with pytest.raises(RuntimeError, match="not unlocked"):                    # right folder, but no unlock
        bt.load_months("2026-08", MONTH, holdout_dir=hd)
    monkeypatch.setenv(UNLOCK, "2026-10")
    with pytest.raises(RuntimeError, match="not unlocked"):                    # another month's unlock
        bt.load_months("2026-08", MONTH, holdout_dir=hd)
    monkeypatch.setenv(UNLOCK, MONTH)
    with pytest.raises(RuntimeError, match="protected holdout"):
        bt.load_months("2026-08", MONTH, holdout_dir=tmp_path / "elsewhere" / "bts")   # wrong folder
    df = bt.load_months("2026-08", MONTH, holdout_dir=hd)
    assert list(df["ym"]) == ["2026-08", "2026-09"] and df["FlightDate"].iloc[-1] == pd.Timestamp("2026-09-04")
    month_frame(MONTH, ["2026-09-04", "2026-10-01"]).to_parquet(hd / "2026-09.parquet")
    with pytest.raises(RuntimeError, match="outside"):
        bt.load_months("2026-08", MONTH, holdout_dir=hd)


def make_sh_tree(tmp_path: Path) -> Path:
    """A copy of fetch_bts.sh with a fake curl and nothing else, so nothing real is touched or downloaded."""
    (tmp_path / "backtest").mkdir()
    shutil.copy(ROOT / "backtest" / "fetch_bts.sh", tmp_path / "backtest" / "fetch_bts.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    zp = bts_zip(tmp_path, "2026-09")
    curl = bin_dir / "curl"
    curl.write_text('#!/bin/sh\nwhile [ $# -gt 0 ]; do case $1 in -o) out=$2; shift;; http*) url=$1;; esac; shift; done\n'
                    f'echo "$url" >> "{tmp_path}/curl.log"\ncp "{zp}" "$out"\n')
    curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
    return bin_dir


SHELLS = [s for s in ("sh", "dash") if shutil.which(s)]


def run_sh(shell, tmp_path, bin_dir, *args, env=None):
    base = {k: v for k, v in os.environ.items() if k != UNLOCK}
    return subprocess.run([shell, str(tmp_path / "backtest" / "fetch_bts.sh"), *args], capture_output=True, text=True,
                          env={**base, "PATH": f"{bin_dir}:/usr/bin:/bin", **(env or {})}, timeout=30)


@pytest.mark.skipif(not shutil.which("unzip"), reason="unzip needed by fetch_bts.sh")
@pytest.mark.parametrize("shell", SHELLS)
def test_fetch_bts_sh_holdout_policy(shell, tmp_path):
    bin_dir = make_sh_tree(tmp_path)
    log = tmp_path / "curl.log"
    cases = [(("--holdout", MONTH), {}),                                    # locked
             (("--holdout", MONTH), {UNLOCK: "1"}),                         # the old value
             (("--holdout", MONTH), {UNLOCK: "2026-10"}),                   # another month
             (("--holdout", "2026-10"), {UNLOCK: MONTH}),                   # another month's flag
             (("--holdout", "2026-08"), {UNLOCK: "2026-08"}),               # not a protected month
             (("--holdout", "2026-10"), {UNLOCK: "2026-10"}),               # protected but not allowlisted
             (("--holdout", "2026-9"), {UNLOCK: MONTH}),                    # malformed
             (("--holdout", MONTH, "2026"), {UNLOCK: MONTH}),               # extra args
             (("2026", "2026", "9"), {UNLOCK: MONTH}),                      # normal form refuses September always
             (("2026", "2026", "9"), {UNLOCK: "1"}),
             (("2026", "2027", "1"), {UNLOCK: MONTH})]
    for args, env in cases:
        r = run_sh(shell, tmp_path, bin_dir, *args, env=env)
        assert r.returncode == 2, (args, env, r.stderr)
        assert not log.exists(), (args, env, "curl was called")
    assert not (tmp_path / "data").exists()
    r = run_sh(shell, tmp_path, bin_dir, "--holdout", MONTH, env={UNLOCK: MONTH})
    assert r.returncode == 0, r.stderr
    assert log.read_text().strip().endswith("_2026_9.zip") and len(log.read_text().split()) == 1
    assert [str(p.relative_to(tmp_path / "data")) for p in (tmp_path / "data").rglob("*.zip")] == [
        "holdout/2026-09/bts/On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_2026_9.zip"]
    assert not (tmp_path / "data" / "bts").exists()                         # the main folder was never created


# ---------------------------------------------------------------- loaders with an extra folder

def test_metar_and_weather_loaders_take_an_extra_folder(tmp_path):
    main, hd = tmp_path / "metar", tmp_path / "hmetar"
    main.mkdir()
    hd.mkdir()
    cols = "station,valid,vsby,sknt,gust,skyc1,skyl1,wxcodes,p01i\n"
    (main / "ATL_2026.csv").write_text(cols + "ATL,2026-08-31 23:53,10,5,,CLR,,,\n")
    (hd / "ATL_2026-09.csv").write_text(cols + "ATL,2026-09-01 00:53,10,5,,CLR,,,\n")
    airports = pd.Index(["ATL"])
    assert len(metar.load_raw(airports, main)) == 1                          # default: main folder only
    both = metar.load_raw(airports, main, [hd])
    assert len(both) == 2 and set(both["station"]) == {"ATL"}
    M = metar.Metar(airports, main, metar_dirs=[hd])
    assert M.max_valid == pd.Timestamp("2026-09-01 00:53")
    assert metar.Metar(airports, main).max_valid == pd.Timestamp("2026-08-31 23:53")

    wm, wh = tmp_path / "w", tmp_path / "wh"
    wm.mkdir()
    wh.mkdir()
    vcols = [f"{v}_previous_day{d}" for v in ("precipitation", "snowfall", "wind_gusts_10m", "cape", "weather_code") for d in (1, 7)]
    for folder, t in ((wm, "2026-08-31 23:00"), (wh, "2026-09-01 00:00")):
        pd.DataFrame({"airport": ["ATL"], "utc_hour": [pd.Timestamp(t)], **{c: [0.0] for c in vcols}}).to_parquet(folder / "ATL.parquet")
    assert weather.Weather(airports, wm).max_utc == pd.Timestamp("2026-08-31 23:00")
    assert weather.Weather(airports, wm, weather_dirs=[wh]).max_utc == pd.Timestamp("2026-09-01 00:00")


def test_b5_1_run_hard_fails_on_holdout_weather(tmp_path):
    wm = tmp_path / "w"
    wm.mkdir()
    vcols = [f"{v}_previous_day{d}" for v in ("precipitation", "snowfall", "wind_gusts_10m", "cape", "weather_code") for d in (1, 7)]
    pd.DataFrame({"airport": ["ATL"], "utc_hour": [pd.Timestamp("2026-09-01 00:00")], **{c: [0.0] for c in vcols}}).to_parquet(wm / "ATL.parquet")
    with pytest.raises(RuntimeError, match="holdout"):
        e.check_weather_holdout(weather.Weather(pd.Index(["ATL"]), wm))


# ---------------------------------------------------------------- experiments_b5 stages

def fake_manifest(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "1d_base.joblib").write_bytes(b"model bytes")
    manifest = {"holdout_month": MONTH, "code": {"commit": "abc"},
                "models": {"1d_base": {"file": "1d_base.joblib", "sha256": e.sha256_file(models / "1d_base.joblib")}}}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))
    return manifest, mp, models


@pytest.fixture
def score_env(tmp_path, hroot, monkeypatch):
    """Everything preflight needs, faked: models, manifest, airports file, git, a passing check marker."""
    manifest, mp, models = fake_manifest(tmp_path)
    manifest["airports_file_sha256"] = e.sha256_file(holdout.AIRPORTS_FILE)
    mp.write_text(json.dumps(manifest))
    res = tmp_path / "results"
    res.mkdir()
    monkeypatch.setenv(UNLOCK, MONTH)
    monkeypatch.setattr(e, "verify_frozen_code", lambda m: None)
    (hroot / MONTH / "metar").mkdir(parents=True)
    (hroot / MONTH / "metar" / "ATL_2026-09.csv").write_text("data")
    e.check_marker_path(MONTH, res).write_text(json.dumps({"files": e.hash_holdout_files(MONTH)}))
    return manifest, mp, models, res


def preflight(env, **kw):
    manifest, mp, models, res = env
    return e.preflight_score(MONTH, results_dir=res, manifest_path=mp, models_dir=models, **kw)


def test_score_preflight_refusals(score_env, hroot, monkeypatch):
    manifest, mp, models, res = score_env
    m, out = preflight(score_env)
    assert m == manifest and out == res / "h1_b5_1_2026-09.json"
    monkeypatch.delenv(UNLOCK)
    with pytest.raises(SystemExit, match="locked"):                          # no unlock
        preflight(score_env)
    monkeypatch.setenv(UNLOCK, MONTH)
    with pytest.raises(SystemExit, match="missing"):                         # no manifest
        e.preflight_score(MONTH, results_dir=res, manifest_path=res / "nope.json", models_dir=models)
    mp.write_text(json.dumps({**manifest, "holdout_month": "2026-10"}))
    with pytest.raises(SystemExit, match="manifest is for"):                 # manifest for another month
        preflight(score_env)
    mp.write_text(json.dumps({**manifest, "airports_file_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="airports"):                        # airport lists changed
        preflight(score_env)
    mp.write_text(json.dumps(manifest))
    (models / "1d_base.joblib").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="manifest"):                        # hash mismatch
        preflight(score_env)
    (models / "1d_base.joblib").write_bytes(b"model bytes")
    (hroot / MONTH / "metar" / "ATL_2026-09.csv").write_text("changed")
    with pytest.raises(SystemExit, match="changed since"):                   # data changed after check passed
        preflight(score_env)
    (hroot / MONTH / "metar" / "ATL_2026-09.csv").write_text("data")
    e.check_marker_path(MONTH, res).unlink()
    with pytest.raises(SystemExit, match="marker"):                          # check never passed
        preflight(score_env)


def test_frozen_code_is_enforced(monkeypatch):
    class R:
        def __init__(self, out="", rc=0):
            self.stdout, self.returncode = out, rc
    state = {"dirty": "", "head": "t1", "freeze": "t1"}

    def fake_git(*a):
        if a[0] == "status":
            return R(state["dirty"])
        return R(state["head"] if a[1].startswith("HEAD:") else state["freeze"])

    monkeypatch.setattr(e, "_git", fake_git)
    manifest = {"code": {"commit": "abcdef0123456789"}}
    e.verify_frozen_code(manifest)
    state["dirty"] = " M flyrely_features/faa.py"
    with pytest.raises(SystemExit, match="uncommitted"):
        e.verify_frozen_code(manifest)
    state.update(dirty="", head="t2")
    with pytest.raises(SystemExit, match="differ from the freeze commit"):
        e.verify_frozen_code(manifest)


def test_attempt_record_and_reruns_never_overwrite(score_env):
    manifest, mp, models, res = score_env
    _, out = preflight(score_env)
    e.record_attempt(MONTH, out, manifest, results_dir=res)                  # written before any data is loaded
    with pytest.raises(SystemExit, match="scored once"):                     # interrupted run: no result file yet
        preflight(score_env)
    with pytest.raises(SystemExit, match="--reason"):
        preflight(score_env, force=True)
    out.write_text('{"original": true}')
    _, r1 = preflight(score_env, force=True, reason="bug in X")
    assert r1.name == "h1_b5_1_2026-09_rerun1.json" and out.read_text() == '{"original": true}'
    e.record_attempt(MONTH, r1, manifest, "bug in X", results_dir=res)
    r1.write_text("{}")
    _, r2 = preflight(score_env, force=True, reason="another")
    assert r2.name.endswith("_rerun2.json")
    recs = json.loads((res / "h1_b5_1_2026-09.attempts.json").read_text())
    assert [r["output"] for r in recs] == ["h1_b5_1_2026-09.json", "h1_b5_1_2026-09_rerun1.json"] and recs[1]["reason"] == "bug in X"


def test_force_rerun_with_no_earlier_attempt_is_refused(score_env):
    with pytest.raises(SystemExit, match="no earlier attempt"):
        preflight(score_env, force=True, reason="x")


def test_score_never_fits_a_model(tmp_path, monkeypatch):
    """Run the whole score stage on tiny stub data with every fit method set to raise."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    import joblib
    import experiments_b5
    rng = np.random.default_rng(0)
    base, a_c, b_c = ["b1", "b2"], ["mt_o_x", "mt_d_x"], ["faa_o_x", "faa_d_x"]
    allc = base + a_c + b_c
    Xt = pd.DataFrame(rng.random((200, 6)), columns=allc)
    yt = (rng.random(200) < 0.3).astype(int)
    models = tmp_path / "models"
    models.mkdir()
    manifest = {"holdout_month": MONTH, "code": {"commit": "x"}, "models": {}, "feature_columns": {}}

    def add(name, cols):
        m = HistGradientBoostingClassifier(max_iter=5).fit(Xt[cols], yt)
        iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(Xt[cols])[:, 1], yt)
        joblib.dump({"model": m, "iso": iso, "columns": cols}, models / f"{name}.joblib")
        manifest["models"][name] = {"file": f"{name}.joblib", "sha256": e.sha256_file(models / f"{name}.joblib")}

    add("booking", ["b1", "b2"])
    for h, sources in e.TESTS.items():
        cols = {"base": base, "A": base + a_c, "B": base + b_c, "AB": base + a_c + b_c}
        manifest["feature_columns"][h] = base + (a_c if "A" in sources else []) + (b_c if "B" in sources else [])
        for arm in ["base"] + [a for a in ("A", "B", "AB") if set(a) <= set(sources)]:
            add(f"{h}_{arm}", cols[arm])
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))
    res = tmp_path / "results"
    res.mkdir()
    n = 300
    ev = pd.DataFrame({"FlightDate": pd.date_range("2026-09-01", periods=30).repeat(10), "cancelled": False,
                       "delayed": (rng.random(n) < 0.2).astype(float), "disrupted": (rng.random(n) < 0.25).astype(float),
                       "ym": MONTH})

    class Stub:
        ok = True
        n_airports = 3

        def features(self, A, rows, h):
            return {c: rng.random(len(rows)) for c in ([c for c in a_c] if self.kind == "A" else b_c)}

    M_, P_ = Stub(), Stub()
    M_.kind, P_.kind = "A", "B"
    M_.max_valid = None
    ctx = {"df": ev, "A": {}, "W": Stub(), "C": None, "R": None, "M": M_, "P": P_,
           "prog": pd.DataFrame({"kind": ["GS"], "status": ["active"]})}
    monkeypatch.setattr(e, "build_context", lambda month: ctx)
    monkeypatch.setattr(e.runup, "horizon_features", lambda A, C, R, W, rows, h, bX, wk: pd.DataFrame(
        rng.random((len(rows), 2)), columns=base))
    monkeypatch.setattr(e.bt, "booking_features", lambda frame, hist: pd.DataFrame(rng.random((len(frame), 2)), columns=base))
    monkeypatch.setattr(e, "MANIFEST", mp)
    monkeypatch.setattr(e, "MODELS_DIR", models)
    monkeypatch.setattr(e, "RESULTS", res)
    monkeypatch.setattr(e, "preflight_score", lambda month, force=False, reason=None: (manifest, res / f"h1_b5_1_{month}.json"))

    def boom(*a, **k):
        raise AssertionError("score must never fit a model")

    for target in (HistGradientBoostingClassifier, IsotonicRegression):
        monkeypatch.setattr(target, "fit", boom)
    monkeypatch.setattr(e.bt, "fit_calibrated", boom)
    e.stage_score(MONTH)
    r = json.loads((res / f"h1_b5_1_{MONTH}.json").read_text())
    assert set(r["gated"]) == {"H1-B-1d", "H1-A-3h", "H1-B-3h", "H1-A-1h", "H1-B-1h"}
    assert all(g["outcome"] in ("confirmed", "not_confirmed", "contradicted") for g in r["gated"].values())
    assert len(json.loads((res / f"h1_b5_1_{MONTH}.attempts.json").read_text())) == 1


def test_freeze_is_refused_when_holdout_data_is_present(hroot, monkeypatch):
    d = hroot / MONTH / "metar"
    d.mkdir(parents=True)
    (d / "ATL_2026-09.csv").write_text("x")
    with pytest.raises(SystemExit, match="holdout data"):
        e.stage_freeze(MONTH)
    shutil.rmtree(hroot / MONTH)
    monkeypatch.setattr(holdout, "locked_files_in_main", lambda *a: ["data/bts/parquet_v2/2026-09.parquet"])
    with pytest.raises(SystemExit, match="holdout data"):
        e.stage_freeze(MONTH)
    with pytest.raises(SystemExit):
        e.stage_freeze("2026-05")                                             # not a protected month


def test_check_and_score_need_the_unlock(hroot):
    with pytest.raises(SystemExit):
        e.stage_check(MONTH)
    with pytest.raises(SystemExit):
        e.stage_score(MONTH)


def test_cli_rules(monkeypatch):
    with pytest.raises(SystemExit):
        e.holdout_cli(["--holdout", MONTH, "--stage", "freeze", "--force-rerun"])
    with pytest.raises(SystemExit):
        e.holdout_cli(["--holdout", MONTH, "--stage", "score", "--reason", "no force flag"])
    with pytest.raises(SystemExit):
        e.holdout_cli(["--stage", "freeze"])                                  # --holdout is required
    with pytest.raises(SystemExit):
        e.holdout_cli(["--holdout", MONTH, "--stage", "nope"])


@pytest.mark.parametrize("diff,ece_with,ece_base,expected", [
    ({"diff": -0.001, "ci95": [-0.002, -0.0005]}, 0.010, 0.008, "confirmed"),
    ({"diff": -0.001, "ci95": [-0.002, -0.0005]}, 0.013, 0.008, "confirmed"),            # ECE +0.005 is the limit
    ({"diff": -0.001, "ci95": [-0.002, 0.0001]}, 0.008, 0.008, "not_confirmed"),         # CI touches 0, point below 0
    ({"diff": 0.0, "ci95": [-0.002, 0.002]}, 0.008, 0.008, "contradicted"),              # point estimate at 0
    ({"diff": 0.0004, "ci95": [-0.001, 0.002]}, 0.008, 0.008, "contradicted"),
    ({"diff": -0.001, "ci95": [-0.002, -0.0005]}, 0.020, 0.008, "contradicted"),         # ECE condition fails
    ({"diff": -0.001, "ci95": [-0.002, 0.0001]}, 0.020, 0.008, "contradicted"),
])
def test_three_outcome_function(diff, ece_with, ece_base, expected):
    assert e.classify_outcome(diff, ece_with, ece_base) == expected
    assert (expected == "confirmed") == e.decide(diff, ece_with, ece_base)["adopt"]       # same rule as B5.1


def test_weekly_brier_diff_is_four_weeks_plus_remainder():
    days = pd.date_range("2026-09-01", "2026-09-30")
    n = len(days)
    ev = pd.DataFrame({"FlightDate": days, "cancelled": False, "delayed": np.zeros(n)})
    weeks = e.weekly_brier_diff(ev, np.full(n, 0.5), np.full(n, 0.4))
    assert len(weeks) == 5 and list(weeks)[0].startswith("wk1 09-01..09-07") and list(weeks)[-1].startswith("wk5 09-29..09-30")
    assert all(abs(v - 0.09) < 1e-9 for v in weeks.values())                              # 0.25 - 0.16


def test_default_b5_1_behaviour_is_untouched():
    assert e.FORBIDDEN_FROM == holdout.LOCKED_FROM_TS
    assert e.TESTS == {"1d": ["B"], "3h": ["A", "B"], "1h": ["A", "B"]}
    assert e.ECE_TOLERANCE == 0.005
