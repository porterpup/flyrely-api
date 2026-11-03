"""Forecast weather features from archived forecasts (Open-Meteo Previous Runs): never observed weather."""

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import DAY0, SHIFT, T, WEATHER_VARS

WEATHER = Path(__file__).resolve().parent.parent / "data" / "weather"


class Weather:
    """Archived forecasts per airport and UTC hour (Open-Meteo Previous Runs)."""

    def __init__(self, airports: pd.Index, weather_dir: Path = None):
        files = sorted((weather_dir or WEATHER).glob("*.parquet"))
        self.ok = bool(files)
        if not self.ok:
            return
        w = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        self.n_airports = w["airport"].nunique()
        ap = airports.get_indexer(w["airport"])
        hour = ((w["utc_hour"] - DAY0) / pd.Timedelta(hours=1)).astype(np.int64).to_numpy()
        keep = ap >= 0
        self.index = pd.Index(ap[keep] * T + hour[keep])
        self.values = w.loc[keep].reset_index(drop=True)

    def features(self, A, rows, lead: int) -> dict:
        out = {}
        for side, ap, t in (("o", A["o_id"][rows], A["s_dep"][rows]), ("d", A["d_id"][rows], A["s_arr"][rows])):
            utc = t - SHIFT + A["utc_shift"][rows]
            hour = np.floor(np.nan_to_num(utc) / 60).astype(np.int64)
            i = self.index.get_indexer(ap * T + hour)
            for v in WEATHER_VARS:
                col = self.values[f"{v}_previous_day{lead}"].to_numpy(dtype="float64")
                out[f"wx_{side}_{v}"] = np.where(i >= 0, col[np.clip(i, 0, None)], np.nan)
        return out


def serve_row(W: Weather, query: dict, lead: int) -> dict:
    """One flight. `query` holds 1-element o_id, d_id, s_dep, s_arr and utc_shift arrays."""
    out = W.features(query, np.array([0]), lead)
    return {k: float(v[0]) for k, v in out.items()}
