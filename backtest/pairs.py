"""
Proxy connections from BTS legs (protocol section 3). Not booked passengers: physically possible, same-network
connections at the top-30 hubs, stratified by layover bucket so tight connections (where misses happen) are
oversampled. Each pair carries a weight `w` (= candidates in its bucket) to reweight to the all-pairs layover mix.

    python backtest/pairs.py [--months 2025-01 2026-08]

Input is the per-month cache from arrival_data.py. Output: data/cache/arrival/pairs_YYYY-MM.parquet, keyed by the
month in which leg 1 is scheduled to arrive. Rows of 2026-09 or later are never loaded.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import arrival_data as ad  # noqa: E402
from flyrely_dist import mct as mctmod  # noqa: E402

SEED = 42
MAX_LAYOVER = 240
MIN_DIRECT_MILES = 150
MAX_DETOUR = 1.6
EXCLUDED_NETWORKS = {"NK"}
HUBS = tuple(mctmod.TOP30)
BUCKETS = ("tight", "short", "medium", "long")


def network_family(net: pd.Series) -> pd.Series:
    """AS and HA are one network (Hawaiian flies as Alaska)."""
    return net.astype(str).replace({"HA": "AS"})


def bucket_edges(m: int) -> list[int]:
    """[M, M+30), [M+30, M+60), [M+60, 120), [120, 240] exactly as pre-registered (inclusive of 240)."""
    return [m, m + 30, m + 60, max(m + 60, 120), MAX_LAYOVER + 1]


def haversine_miles(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def _coords():
    ref = pd.read_csv(ad.AIRPORTS_REF)
    return {r.iata: (r.lat, r.lon) for r in ref.itertuples()}


def _gc(coords, a: pd.Series, b: pd.Series) -> np.ndarray:
    la = a.map(lambda x: coords.get(x, (np.nan, np.nan))[0]).to_numpy(float)
    oa = a.map(lambda x: coords.get(x, (np.nan, np.nan))[1]).to_numpy(float)
    lb = b.map(lambda x: coords.get(x, (np.nan, np.nan))[0]).to_numpy(float)
    ob = b.map(lambda x: coords.get(x, (np.nan, np.nan))[1]).to_numpy(float)
    return haversine_miles(la, oa, lb, ob)


def prepare_legs(legs: pd.DataFrame) -> pd.DataFrame:
    """Add row ids, network family and arrival date; legs must be the cache schema (arrival_data.build_cache_frame)."""
    L = legs.reset_index(drop=True).copy()
    L["row"] = np.arange(len(L))
    L["fam"] = network_family(L["network"])
    L["arr_date"] = pd.to_datetime(L["arr_wall"]).dt.normalize()
    L["dep_date"] = pd.to_datetime(L["flight_date"]).dt.normalize()
    return L


def _is_through(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    same_op = (a["carrier"].to_numpy() == b["carrier"].to_numpy()) & \
              (a["fn_operating"].to_numpy() == b["fn_operating"].to_numpy())
    same_mk = (a["fam"].to_numpy() == b["fam"].to_numpy()) & \
              (a["fn_marketing"].to_numpy() == b["fn_marketing"].to_numpy())
    return same_op | same_mk


def candidate_pairs(L1: pd.DataFrame, L2: pd.DataFrame, hub: str, m: int, coords: dict) -> pd.DataFrame:
    """All eligible (leg 1, leg 2) rows for one hub and one set of dates (frames from `prepare_legs`)."""
    a = L1[(L1["dest"] == hub) & ~L1["fam"].isin(EXCLUDED_NETWORKS) & L1["arr_wall"].notna()]
    b = L2[(L2["origin"] == hub) & ~L2["fam"].isin(EXCLUDED_NETWORKS) & L2["dep_wall"].notna()]
    if a.empty or b.empty:
        return pd.DataFrame()
    keep = ["row", "fam", "arr_date", "dep_date", "arr_wall", "dep_wall", "origin", "dest", "carrier", "fn_operating",
            "fn_marketing", "tail", "arr_delay", "dep_delay", "cancelled", "diverted"]
    a, b = a[keep], b[keep]
    parts = []
    for day, ad_ in a.groupby("arr_date"):          # one day at a time keeps the join small at the biggest hubs
        bd = b[b["dep_date"] == day]
        if bd.empty:
            continue
        j = ad_.merge(bd, on="fam", suffixes=("1", "2"))
        lay = ((pd.to_datetime(j["dep_wall2"]) - pd.to_datetime(j["arr_wall1"])).dt.total_seconds() // 60).astype(int)
        j["layover"] = lay
        parts.append(j[(lay >= m) & (lay <= MAX_LAYOVER)])
    if not parts:
        return pd.DataFrame()
    j = pd.concat(parts, ignore_index=True)
    if j.empty:
        return j
    j = j[j["dest2"] != j["origin1"]]
    if j.empty:
        return j
    d1 = _gc(coords, j["origin1"], j["dest1"])
    d2 = _gc(coords, j["origin2"], j["dest2"])
    d = _gc(coords, j["origin1"], j["dest2"])
    ok = (d >= MIN_DIRECT_MILES) & ((d1 + d2) / np.where(d > 0, d, np.nan) <= MAX_DETOUR)
    j = j[ok]
    if j.empty:
        return j
    a1 = j[[c + "1" for c in ("carrier", "fn_operating", "fn_marketing")] + ["fam"]]
    a1.columns = ["carrier", "fn_operating", "fn_marketing", "fam"]
    b2 = j[[c + "2" for c in ("carrier", "fn_operating", "fn_marketing")] + ["fam"]]
    b2.columns = ["carrier", "fn_operating", "fn_marketing", "fam"]
    through = _is_through(a1, b2)
    tail_same = (j["tail1"].to_numpy() == j["tail2"].to_numpy()) & (j["tail1"].to_numpy() != "")
    return j[~through & ~tail_same]


def outcomes(x1, y2, c1, c2, div1, layover, m):
    """miss: both legs operated and X - Y > s (s = layover - M); a diverted leg 1 is a miss. X - Y == s is not."""
    s = layover - m
    both = ~c1 & ~c2
    late = div1 | (np.nan_to_num(x1, nan=0.0) - np.nan_to_num(y2, nan=0.0) > s)
    miss = both & late
    return miss, miss | c1 | c2


def build_pairs(legs1: pd.DataFrame, legs2: pd.DataFrame, mct: dict, hubs=HUBS, seed: int = SEED,
                coords: dict = None, max_date="2026-08-31") -> pd.DataFrame:
    """Stratified proxy pairs. `legs1` are the legs whose scheduled arrival falls in the target month; `legs2`
    the legs that may depart on those dates (same month and the next). Rows dated after `max_date` are refused."""
    coords = coords or _coords()
    L1, L2 = prepare_legs(legs1), prepare_legs(legs2)
    cap = pd.Timestamp(max_date)
    for L in (L1, L2):
        if len(L) and (L["dep_date"].max() > cap):
            raise ValueError("legs dated after the protocol window were passed to build_pairs")
    L1 = L1[L1["arr_date"] <= cap]
    out = []
    for hi, hub in enumerate(hubs):
        m = int(mct.get(hub, mctmod.DEFAULT_MCT))
        c = candidate_pairs(L1, L2, hub, m, coords)
        if c.empty:
            continue
        edges = np.array(bucket_edges(m))
        c["bucket"] = np.searchsorted(edges, c["layover"].to_numpy(), side="right") - 1
        rng = np.random.default_rng([seed, hi])
        c["u"] = rng.random(len(c))
        c = c.sort_values(["row1", "bucket", "u"], kind="stable")
        g = c.groupby(["row1", "bucket"], sort=False)
        n_cand = g["u"].transform("size")
        pick = c[g.cumcount() == 0].copy()
        pick["w"] = n_cand[pick.index].to_numpy()
        pick["hub"], pick["M"] = hub, m
        out.append(pick)
    if not out:
        return pd.DataFrame()
    P = pd.concat(out, ignore_index=True)
    c1, c2 = P["cancelled1"].to_numpy(bool), P["cancelled2"].to_numpy(bool)
    miss, disrupted = outcomes(P["arr_delay1"].to_numpy(float), P["dep_delay2"].to_numpy(float), c1, c2,
                               P["diverted1"].to_numpy(bool), P["layover"].to_numpy(), P["M"].to_numpy())
    res = pd.DataFrame({
        "ym": P["arr_date1"].dt.strftime("%Y-%m"), "arr_date": P["arr_date1"], "hub": P["hub"], "network": P["fam"],
        "layover": P["layover"].astype("int16"), "M": P["M"].astype("int16"), "bucket": P["bucket"].astype("int8"),
        "w": P["w"].astype("float32"), "row1": P["row1"].astype("int32"), "row2": P["row2"].astype("int32"),
        "x": P["arr_delay1"].astype("float32"), "y": P["dep_delay2"].astype("float32"),
        "c1": c1, "c2": c2, "div1": P["diverted1"].to_numpy(bool), "miss": miss, "disrupted": disrupted,
    })
    return res


def build_month_pairs(ym: str, cache_dir: Path = None, force: bool = False) -> Path:
    """Pairs whose leg 1 arrives in `ym`; leg 2 comes from `ym` and the following month if it is in the window."""
    ad.guard_month(ym)
    cache_dir = Path(cache_dir or ad.CACHE_DIR)
    out = cache_dir / f"pairs_{ym}.parquet"
    if out.exists() and not force:
        return out
    cur = ad.load_cache(ym, cache_dir)
    nxt_ym = f"{(pd.Period(ym) + 1)}"
    frames = [cur]
    if nxt_ym <= ad.LAST_MONTH:
        frames.append(ad.load_cache(nxt_ym, cache_dir))
    legs2 = pd.concat(frames, ignore_index=True)
    # row ids refer to `legs2`/`cur` positions; keep a month tag so the evaluator can find the legs again
    tab = mctmod.load_table(ad.ROOT / "models" / "mct_defaults.csv")
    pairs = build_pairs(cur, legs2, tab)
    pairs = pairs[pairs["ym"] == ym].reset_index(drop=True)
    pairs["ym2"] = np.where(legs2["ym"].to_numpy()[pairs["row2"].to_numpy()] == ym, ym, nxt_ym)
    pairs["row2"] = np.where(pairs["ym2"] == ym, pairs["row2"], pairs["row2"] - len(cur)).astype("int32")
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out, index=False, compression="zstd")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", nargs=2, default=[ad.FIRST_MONTH, ad.LAST_MONTH])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    for ym in ad.months(*args.months):
        p = build_month_pairs(ym, force=args.force)
        d = pd.read_parquet(p, columns=["miss", "w"])
        print(p.name, len(d), f"miss(unweighted)={d['miss'].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
