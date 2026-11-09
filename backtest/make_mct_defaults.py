"""Write models/mct_defaults.csv from 2025 BTS departures (parquet_v2, 2025 months only)."""

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flyrely_dist import mct  # noqa: E402


def main():
    counts = pd.Series(dtype=float)
    for m in range(1, 13):
        t = pq.read_table(ROOT / "data" / "bts" / "parquet_v2" / f"2025-{m:02d}.parquet", columns=["Origin"])
        counts = counts.add(t.to_pandas()["Origin"].astype(str).value_counts(), fill_value=0)
    top = [a for a in counts.sort_values(ascending=False).index[:30]]
    assert top == mct.TOP30, "2025 top-30 differs from the frozen list"
    out = mct.build_defaults(counts)
    out.to_csv(ROOT / "models" / "mct_defaults.csv", index=False)
    print(out["class"].value_counts().to_dict())


if __name__ == "__main__":
    main()
