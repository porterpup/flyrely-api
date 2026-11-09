"""Default minimum connect times (MCT), frozen in backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md (D1, approved 2026-11-09).

No free complete MCT source exists, so these are FlyRely estimates by airport size, labelled as such.
"""

from __future__ import annotations

import pandas as pd

# Frozen from 2025 BTS departures (parquet_v2 2025-01..12), largest first.
TOP30 = ("ORD ATL DEN DFW CLT PHX LAX LAS SEA IAH LGA MCO SFO DCA BOS DTW EWR MSP JFK PHL SLC MIA BNA SAN BWI IAD "
         "AUS FLL TPA PDX").split()
LARGE_CAMPUS = "ATL ORD DFW DEN LAX JFK EWR IAH SFO CLT".split()
SOURCE = "flyrely_default_estimate"
DEFAULT_MCT = 30            # airports absent from the table


def classify(airport: str, deps_per_day: float) -> tuple[int, str]:
    """(minutes, class) for one airport. The large-campus list wins over the top-30 rule."""
    if airport in LARGE_CAMPUS:
        return 50, "large_campus"
    if airport in TOP30:
        return 40, "top30"
    if deps_per_day >= 100:
        return 35, "mid"
    return 30, "small"


def build_defaults(departures_2025: pd.Series) -> pd.DataFrame:
    """Table from a Series of 2025 departure counts indexed by airport code."""
    rows = []
    for ap, n in departures_2025.sort_values(ascending=False).items():
        per_day = float(n) / 365.0
        minutes, cls = classify(str(ap), per_day)
        rows.append({"airport": ap, "deps_per_day_2025": round(per_day, 1), "mct_minutes": minutes,
                     "class": cls, "source": SOURCE})
    return pd.DataFrame(rows)


def load_table(path) -> dict[str, int]:
    df = pd.read_csv(path)
    return dict(zip(df["airport"], df["mct_minutes"].astype(int)))
