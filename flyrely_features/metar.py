"""Observed airport weather (METAR) at the cutoff: the latest report issued at or before the cutoff.

Training reads the Iowa Environmental Mesonet ASOS archive (``backtest/fetch_metar.py``). Serving reads the same
fields live from aviationweather.gov (``from_aviationweather``). Only reports timestamped at or before the
cutoff are used, and a report older than ``MAX_AGE`` minutes counts as missing (NaN), so a stale station never
masquerades as "calm".
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import DAY0, SHIFT, T

METAR_DIR = Path(__file__).resolve().parent.parent / "data" / "metar"
MAX_AGE = 180            # minutes; an older report is treated as missing
NO_CEILING = 12000.0     # feet; value for "no broken/overcast layer"
FEATURES = ["ceiling_ft", "vis_mi", "wind_kt", "gust_kt", "precip", "snow", "thunder", "freezing", "fog",
            "ifr", "p01i", "age_min"]
_PRECIP = re.compile(r"(RA|DZ|SN|GR|GS|PL|SG|IC|UP)")


def _tokens(wx) -> list:
    return str(wx).split() if isinstance(wx, str) else []


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Raw IEM-style columns -> numeric weather columns plus ``t`` (minutes since 2024-01-01 00:00 UTC).

    Needs: valid (UTC), vsby, sknt, gust, skyc1..skyc4, skyl1..skyl4, wxcodes, p01i."""
    out = pd.DataFrame(index=df.index)
    out["t"] = ((pd.to_datetime(df["valid"]) - DAY0) / pd.Timedelta(minutes=1)).astype("float64")
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
    ceil = np.full(len(df), NO_CEILING)
    for k in "1234":
        cov = df[f"skyc{k}"].astype(str) if f"skyc{k}" in df else pd.Series("", index=df.index)
        base = num(f"skyl{k}").to_numpy()
        is_ceiling = cov.isin(["BKN", "OVC", "VV"]).to_numpy() & ~np.isnan(base)
        ceil = np.where(is_ceiling, np.minimum(ceil, base), ceil)
    out["ceiling_ft"] = ceil
    out["vis_mi"] = num("vsby")
    out["wind_kt"] = num("sknt")
    out["gust_kt"] = num("gust").fillna(0)
    toks = [_tokens(w) for w in df["wxcodes"]] if "wxcodes" in df else [[] for _ in range(len(df))]
    has = lambda f: np.array([any(f(t) for t in ts) for ts in toks], dtype="float64")
    out["precip"] = has(lambda t: bool(_PRECIP.search(t.lstrip("+-"))) and not t.startswith("VC"))
    out["snow"] = has(lambda t: ("SN" in t or "SG" in t) and not t.startswith("VC"))
    out["thunder"] = has(lambda t: "TS" in t)       # includes VCTS: thunderstorms in the vicinity stop ramps
    out["freezing"] = has(lambda t: "FZ" in t)
    out["fog"] = has(lambda t: "FG" in t and not t.startswith("VC"))
    out["ifr"] = ((out["ceiling_ft"] < 1000) | (out["vis_mi"] < 3)).astype("float64")
    out["p01i"] = num("p01i").fillna(0)
    return out


def from_aviationweather(records: list) -> pd.DataFrame:
    """Rows of the aviationweather.gov METAR JSON (``/api/data/metar?ids=KATL&format=json``) -> IEM-style raw columns.

    Fields used: icaoId, obsTime (epoch seconds), visib ("10+" or number), wspd, wgst, wxString, clouds [{cover, base}]."""
    rows = []
    for r in records:
        row = {"station": str(r.get("icaoId", ""))[-3:], "valid": pd.to_datetime(r["obsTime"], unit="s"),
               "vsby": str(r.get("visib", "")).replace("+", "") or np.nan, "sknt": r.get("wspd"),
               "gust": r.get("wgst"), "wxcodes": r.get("wxString"), "p01i": r.get("precip")}
        for k, layer in enumerate((r.get("clouds") or [])[:4], 1):
            row[f"skyc{k}"], row[f"skyl{k}"] = layer.get("cover"), layer.get("base")
        rows.append(row)
    return pd.DataFrame(rows)


def load_raw(airports: pd.Index, metar_dir: Path = None) -> pd.DataFrame:
    frames = []
    for f in sorted((metar_dir or METAR_DIR).glob("*.csv")):
        d = pd.read_csv(f, dtype=str)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class Metar:
    """Every report per airport, queryable at any cutoff time (minutes since 2024-01-01 00:00 UTC)."""

    def __init__(self, airports: pd.Index, metar_dir: Path = None, raw: pd.DataFrame = None):
        raw = raw if raw is not None else load_raw(airports, metar_dir)
        self.ok = len(raw) > 0
        if not self.ok:
            return
        d = derive(raw)
        ap = airports.get_indexer(raw["station"])
        keep = (ap >= 0) & d["t"].notna().to_numpy()
        d, ap = d[keep], ap[keep]
        key = ap.astype(np.int64) * T + np.floor(d["t"].to_numpy()).astype(np.int64)
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.t = d["t"].to_numpy()[order]
        self.cols = {c: d[c].to_numpy(dtype="float64")[order] for c in FEATURES if c != "age_min"}
        self.covered = np.zeros(len(airports), dtype=bool)
        self.covered[np.unique(ap)] = True
        self.n_airports = int(self.covered.sum())

    def at(self, ap: np.ndarray, c: np.ndarray) -> dict:
        """Latest report at or before c for airport ids `ap`; NaN where uncovered, none yet or older than MAX_AGE."""
        c = np.asarray(c, dtype="float64")
        bad = (ap < 0) | np.isnan(c)
        covered = np.where(ap >= 0, self.covered[np.clip(ap, 0, None)], False) & ~bad
        q = np.where(covered, ap * T + np.floor(np.nan_to_num(c)).astype(np.int64), 0)
        j = np.searchsorted(self.key, q, side="right") - 1
        jj = np.clip(j, 0, len(self.key) - 1)
        age = c - self.t[jj]
        ok = covered & (j >= 0) & (self.key[jj] // T == ap) & (age <= MAX_AGE)
        out = {k: np.where(ok, v[jj], np.nan) for k, v in self.cols.items()}
        out["age_min"] = np.where(ok, age, np.nan)
        return out

    def features(self, A, rows, h_minutes: int) -> dict:
        """Weather now (at the cutoff = scheduled departure - horizon) at origin and destination."""
        c = A["s_dep"][rows] - h_minutes - SHIFT + A["utc_shift"][rows]
        out = {}
        for side, ap in (("o", A["o_id"][rows]), ("d", A["d_id"][rows])):
            for k, v in self.at(ap, c).items():
                out[f"mt_{side}_{k}"] = v
        return out


def serve_row(M: Metar, query: dict, h_minutes: int) -> dict:
    """One flight. `query` holds 1-element o_id, d_id, s_dep and utc_shift arrays. Same code path as the vectorised call."""
    out = M.features(query, np.array([0]), h_minutes)
    return {k: float(v[0]) for k, v in out.items()}
