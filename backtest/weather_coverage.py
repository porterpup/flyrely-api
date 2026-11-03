"""Share of flights with archived weather for both origin and destination, per validation window.

    python backtest/weather_coverage.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import BTS_PARQUET, ROOT  # noqa: E402

have = {p.stem for p in (ROOT / "data" / "weather").glob("*.parquet")}
for label, months in (("2025-07..2025-12", pd.period_range("2025-07", "2025-12", freq="M")),
                      ("2026-01..2026-08", pd.period_range("2026-01", "2026-08", freq="M"))):
    df = pd.concat([pd.read_parquet(BTS_PARQUET / f"{m}.parquet", columns=["Origin", "Dest"]) for m in months])
    ok = df["Origin"].astype(str).isin(have) & df["Dest"].astype(str).isin(have)
    print(f"{label}: {ok.mean():.2%} of {len(df):,} flights have weather at both ends ({len(have)} airports)")
