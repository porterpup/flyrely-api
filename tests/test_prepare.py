"""BTS Marketing Carrier -> Parquet column mapping and the holdout download lock."""

import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
import prepare  # noqa: E402

# BTS pads some header names with trailing spaces ("Operating_Airline ")
HEADER = ('"FlightDate","Marketing_Airline_Network","Flight_Number_Marketing_Airline","Operating_Airline ",'
          '"Tail_Number","Flight_Number_Operating_Airline","Origin","Dest","CRSDepTime","DepDelay",'
          '"TaxiOut","WheelsOff","CRSArrTime","Cancelled","CancellationCode","Diverted","CRSElapsedTime","Distance","Junk",')
ROWS = [
    '2026-08-03,DL,4632,9E,N123,4632,FWA,ATL,0630,22.0,14.0,0655,0830,0.0,,0.0,95.0,508.0,x,',
    '2026-08-03,AA,5000,PT,N456,5000,CLT,RDU,1015,-3.0,11.0,1030,1130,0.0,,0.0,50.0,130.0,x,',
]


def _zip(tmp_path, name="On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_2026_8.zip"):
    zp = tmp_path / name
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("data.csv", "\n".join([HEADER, *ROWS]) + "\n")
        z.writestr("readme.html", "<html/>")
    return zp


def test_marketing_zip_maps_operating_carrier_to_reporting_columns(tmp_path):
    out = prepare.convert(_zip(tmp_path), tmp_path / "v2", source="marketing")
    assert out.name == "2026-08.parquet"
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert list(df["Reporting_Airline"].astype(str)) == ["9E", "PT"]            # operating, not marketing
    assert list(df["Marketing_Airline_Network"].astype(str)) == ["DL", "AA"]
    assert list(df["Flight_Number_Reporting_Airline"]) == [4632, 5000]
    assert list(df["Flight_Number_Marketing_Airline"]) == [4632, 5000]
    assert {"WheelsOff", "TaxiOut"} <= set(df.columns)
    assert "Junk" not in df.columns and "Operating_Airline" not in df.columns
    assert not any(c != c.strip() for c in df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["FlightDate"])


def _fetch(*args, env=None):
    import os
    return subprocess.run(["sh", str(ROOT / "backtest" / "fetch_bts.sh"), *args], capture_output=True, text=True,
                          env={**os.environ, "PATH": "/nonexistent:/usr/bin:/bin", **(env or {})}, timeout=30)


def test_holdout_lock_refuses_before_any_download():
    for args in (("2026", "2026", "9"), ("2026", "2026", "12"), ("2026", "2027", "1")):
        r = _fetch(*args)
        assert r.returncode == 2, (args, r.stderr)
        assert "holdout" in r.stderr
        assert "fetching" not in r.stdout        # exited before the download loop
