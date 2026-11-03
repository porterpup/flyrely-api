"""Single source of truth for which BTS Parquet directory the backtests read."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# All-carrier data built by `prepare.py --source marketing` (includes 9E, PT, QX, G7, C5, YV).
# The older data/bts/parquet (Reporting Carrier file) omits those six operators.
BTS_PARQUET = Path(os.environ.get("FLYRELY_BTS_PARQUET", ROOT / "data" / "bts" / "parquet_v2"))
