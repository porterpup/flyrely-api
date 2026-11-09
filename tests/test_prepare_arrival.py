"""prepare_arrival: arrival columns added, shared columns equal prepare.py's, holdout months refused."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
import prepare  # noqa: E402
import prepare_arrival  # noqa: E402

HEADER = ('"FlightDate","Marketing_Airline_Network","Flight_Number_Marketing_Airline","Operating_Airline ",'
          '"Tail_Number","Flight_Number_Operating_Airline","Origin","Dest","CRSDepTime","DepDelay",'
          '"TaxiOut","WheelsOff","WheelsOn","TaxiIn","CRSArrTime","ArrTime","ArrDelay","ArrDel15","Cancelled",'
          '"CancellationCode","Diverted","CRSElapsedTime","Distance","DivReachedDest","DivArrDelay","Junk",')
ROWS = [
    '2026-08-03,DL,4632,9E,N123,4632,FWA,ATL,0630,22.0,14.0,0655,0820,6.0,0830,0826,-4.0,0.0,0.0,,0.0,95.0,508.0,,,x,',
    '2026-08-03,AA,5000,PT,N456,5000,CLT,RDU,1015,-3.0,11.0,1030,1120,5.0,1130,1125,-5.0,0.0,0.0,,0.0,50.0,130.0,,,x,',
    '2026-08-03,UA,100,UA,N789,100,SFO,DEN,0800,,,,,,1100,,,,1.0,B,0.0,150.0,967.0,,,x,',
    '2026-08-03,UA,101,UA,N790,101,SFO,ORD,0900,10.0,15.0,0930,,,1500,,,,0.0,,1.0,230.0,1846.0,1.0,95.0,x,',
]


def _zip(tmp_path, year=2026, month=8):
    zp = tmp_path / f"On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("data.csv", "\n".join([HEADER, *ROWS]) + "\n")
    return zp


def test_arrival_columns_added_and_shared_columns_equal_prepare(tmp_path):
    zp = _zip(tmp_path)
    base = pd.read_parquet(prepare.convert(zp, tmp_path / "v2", source="marketing"))
    arr = pd.read_parquet(prepare_arrival.convert(zp, tmp_path / "arr"))
    assert set(prepare_arrival.ARRIVAL_COLUMNS) <= set(arr.columns)
    assert list(arr.columns[:len(base.columns)]) == list(base.columns) or set(base.columns) <= set(arr.columns)
    pd.testing.assert_frame_equal(arr[list(base.columns)], base)
    assert arr["ArrDelay"].iloc[0] == -4.0
    assert pd.isna(arr["ArrDelay"].iloc[2])                      # cancelled
    assert arr["DivArrDelay"].iloc[3] == 95.0 and arr["DivReachedDest"].iloc[3] == 1.0
    assert "Junk" not in arr.columns


def test_locked_month_zip_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        prepare_arrival.convert(_zip(tmp_path, 2026, 9), tmp_path / "arr")
    assert not (tmp_path / "arr").exists()


def test_exits_when_unlock_variable_present_even_for_open_month(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYRELY_HOLDOUT_UNLOCK", "2026-09")
    with pytest.raises(SystemExit):
        prepare_arrival.convert(_zip(tmp_path), tmp_path / "arr")
    with pytest.raises(SystemExit):
        prepare_arrival.main([])


def test_script_has_no_holdout_flag():
    r = subprocess.run([sys.executable, str(ROOT / "backtest" / "prepare_arrival.py"), "--holdout", "2026-09"],
                       capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                            if k != "FLYRELY_HOLDOUT_UNLOCK"})
    assert r.returncode != 0 and "unrecognized" in r.stderr
