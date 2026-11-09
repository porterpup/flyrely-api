"""
Arrival-ladder and missed-connection evaluation, implementing backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md exactly
(including its "Clarifications" section). Run ONCE for the full evaluation; any re-run after a bug fix is an amendment.

    python backtest/eval_arrival_connections.py --quick   # 2025 months only, smoke test (never a result)
    python backtest/eval_arrival_connections.py           # the registered evaluation (2026-01..08)

Reads only the monthly caches, proxy-pair files and the ladder artifact. Nothing dated 2026-09 or later is loaded
(`assert_window`), FLYRELY_HOLDOUT_UNLOCK must not be set. Writes backtest/results/arrival_connections.json
(`_quick` suffix in quick mode) and, in the full run, backtest/REPORT_ARRIVAL_CONNECTIONS.md (draft tables).
Memory: ladders are predicted month by month and cached as float32; pairs stream in chunks (peak well under 6 GB).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import arrival_data as ad  # noqa: E402
import pairs as pr  # noqa: E402
from flyrely_dist import connections as C  # noqa: E402
from flyrely_dist import ladder as L  # noqa: E402
from flyrely_dist import mct as mctmod  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
ARTIFACT = ROOT / "models" / "arrival_ladder_v1.joblib"
SEED = 42
N_BOOT = 1000
TIERS = (0.20, 0.30)
BANDS = (0.05, 0.15)
EVAL_THRESHOLDS = (0, 15, 45, 120)
FIT_MONTHS = ("2025-01", "2025-06")

FULL = dict(name="full", val=("2026-01", "2026-08"), select=("2026-01", "2026-04"), confirm=("2026-05", "2026-08"),
            rho_fit=("2025-07", "2025-09"), rho_sel=("2025-10", "2025-12"), frac=1.0, out="arrival_connections.json")
QUICK = dict(name="quick", val=("2025-11", "2025-12"), select=("2025-11", "2025-11"), confirm=("2025-12", "2025-12"),
             rho_fit=("2025-07", "2025-07"), rho_sel=("2025-10", "2025-10"), frac=0.05,
             out="arrival_connections_quick.json")


# ------------------------------------------------------------------ guards

def assert_window(months):
    """Every month must be inside 2025-01..2026-08; nothing from the protected holdout is ever loaded."""
    if ad.UNLOCK_VAR in os.environ:
        sys.exit(f"{ad.UNLOCK_VAR} is set; the evaluation never runs with it")
    for ym in months:
        if not (ad.FIRST_MONTH <= ym <= ad.LAST_MONTH):
            raise ValueError(f"{ym} is outside the protocol window {ad.FIRST_MONTH}..{ad.LAST_MONTH}")


def months_between(a, b):
    return ad.months(a, b)


# ------------------------------------------------------------------ bootstrap and metrics

def day_block_ci(day, num, den=None, n_boot=N_BOOT, seed=SEED, kind="mean"):
    """95% day-block percentile CI. `day` integer day id per row, `num` per-row values.
    kind "mean": mean(num); "ratio": sum(num)/sum(den); "skill": 1 - sum(num)/sum(den)."""
    day = np.asarray(day)
    _, inv = np.unique(day, return_inverse=True)
    nd = inv.max() + 1
    sn = np.bincount(inv, weights=np.asarray(num, dtype=float), minlength=nd)
    sd = np.bincount(inv, weights=np.ones(len(inv)) if den is None else np.asarray(den, dtype=float), minlength=nd)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nd, (n_boot, nd))
    r = sn[idx].sum(1) / sd[idx].sum(1)
    point = sn.sum() / sd.sum()
    if kind == "skill":
        r, point = 1.0 - r, 1.0 - point
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def deciles(y, p):
    order = np.argsort(p, kind="stable")
    rows = []
    ece = 0.0
    for i, b in enumerate(np.array_split(order, 10)):
        pred, obs = float(p[b].mean()), float(y[b].mean())
        ece += len(b) / len(p) * abs(pred - obs)
        rows.append({"decile": i + 1, "n": int(len(b)), "predicted": pred, "observed": obs})
    return ece, rows


def weighted_ece(y, p, w):
    order = np.argsort(p, kind="stable")
    cw = np.cumsum(w[order])
    cut = np.minimum((cw / cw[-1] * 10).astype(int), 9)
    ece = 0.0
    for b in range(10):
        m = order[cut == b]
        if len(m):
            ece += w[m].sum() / w.sum() * abs(np.average(p[m], weights=w[m]) - np.average(y[m], weights=w[m]))
    return float(ece)


def binary_metrics(y, p, full=True):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = {"n": int(len(y)), "base_rate": float(y.mean()), "mean_pred": float(p.mean()),
           "brier": float(brier_score_loss(y, p))}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
    if full:
        ece, dec = deciles(y, p)
        gaps = [abs(d["predicted"] - d["observed"]) for d in dec]
        out.update({"ece": float(ece), "max_decile_gap": float(max(gaps)), "top_decile_gap": float(gaps[-1]),
                    "max_gap_deciles_1_9": float(max(gaps[:-1])), "deciles": dec,
                    "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))})
    return out


def tier_of(p15):
    return np.where(p15 < TIERS[0], 0, np.where(p15 < TIERS[1], 1, 2))


def band_of(p):
    return np.where(p < BANDS[0], 0, np.where(p < BANDS[1], 1, 2))


def ordered_rates(rates, strict=False):
    """rates: observed rate per ordered group, None for empty groups. Non-decreasing over the non-empty ones with the
    lowest strictly below the highest (C6); strict=True needs every step to rise (tiers, rule A3)."""
    r = [x for x in rates if x is not None]
    if len(r) < 2:
        return True
    if strict:
        return all(a < b for a, b in zip(r, r[1:]))
    return all(a <= b for a, b in zip(r, r[1:])) and r[0] < r[-1]


# ------------------------------------------------------------------ data

CACHE_COLS = ["ym", "flight_date", "network", "carrier", "cancelled", "diverted", "arr_delay", "dep_delay",
              "dep_wall"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Store:
    """Loads monthly caches and ladder predictions (predicting and caching them on first use)."""

    def __init__(self, bundle, names, prod_model, sha):
        self.bundle, self.names, self.prod, self.sha = bundle, names, prod_model, sha
        self._meta, self._pred = {}, {}

    def meta(self, ym):
        assert_window([ym])
        if ym not in self._meta:
            if len(self._meta) >= 4:
                self._meta.pop(next(iter(self._meta)))
            self._meta[ym] = ad.load_cache(ym)[CACHE_COLS]
        return self._meta[ym]

    def pred(self, ym):
        assert_window([ym])
        if ym in self._pred:
            return self._pred[ym]
        path = ad.CACHE_DIR / f"pred_{ym}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            if str(z["sha"]) == self.sha:
                d = {k: z[k] for k in ("arr", "dep", "cancel", "p_prod", "rearr")}
                return self._keep(ym, d)
        c = ad.load_cache(ym)
        X = c[self.names].to_numpy(dtype=np.float32)
        raw_arr = np.stack([m.predict(X) for m in self.bundle.arr], axis=1)
        raw_dep = np.stack([m.predict(X) for m in self.bundle.dep], axis=1)
        d = {"arr": L.rearrange(raw_arr).astype(np.float32), "dep": L.rearrange(raw_dep).astype(np.float32),
             "cancel": self.bundle.cancel.predict(X).astype(np.float32),
             "p_prod": self.prod.predict_proba(c[self.names])[:, 1].astype(np.float32),
             "rearr": L.needs_rearrangement(raw_arr)}
        np.savez(path, sha=self.sha, **d)
        return self._keep(ym, d)

    def _keep(self, ym, d):
        if len(self._pred) >= 4:
            self._pred.pop(next(iter(self._pred)))
        self._pred[ym] = d
        return d


def flown_arrival_rows(meta):
    """Rows scored for arrival: flown flights with a recorded ArrDelay, or diverted (counted late)."""
    can = meta["cancelled"].to_numpy(bool)
    a = meta["arr_delay"].to_numpy(float)
    return ~can & ~np.isnan(a)


def collect_window(store, months):
    """Concatenate predictions and labels of flown flights over months."""
    parts = {k: [] for k in ("P", "dep", "p_prod", "y", "ym", "carrier", "day", "rearr")}
    for ym in months:
        m, p = store.meta(ym), store.pred(ym)
        ok = flown_arrival_rows(m)
        parts["P"].append(p["arr"][ok])
        parts["dep"].append(p["dep"][ok])
        parts["p_prod"].append(p["p_prod"][ok])
        parts["rearr"].append(p["rearr"][ok])
        parts["y"].append(m["arr_delay"].to_numpy(float)[ok])
        parts["ym"].append(np.full(ok.sum(), ym))
        parts["carrier"].append(m["carrier"].to_numpy()[ok])
        parts["day"].append(pd.to_datetime(m["flight_date"]).to_numpy()[ok].astype("datetime64[D]").astype(int))
    return {k: np.concatenate(v) for k, v in parts.items()}


# ------------------------------------------------------------------ arrival

def recalibrated_arm(store, val_months):
    """Arm L-R: for month m, per-threshold logistic recalibration fitted on months m-3, m-4, m-5 (K = 3,
    published by then), applied to month m. Returns {ym: (P_lr, coefficients)} using each month's rows."""
    out = {}
    for m in val_months:
        use = [str(pd.Period(m, "M") - 3 - i) for i in range(3)]
        assert_window(use)
        hist = collect_window(store, use)
        cur = collect_window(store, [m])
        P = np.empty_like(cur["P"], dtype=float)
        coefs = []
        for k, t in enumerate(L.ARR_THRESHOLDS):
            a, b = L.fit_logistic_recal(hist["P"][:, k], (hist["y"] >= t).astype(int))
            P[:, k] = L.apply_logistic_recal(cur["P"][:, k], a, b)
            coefs.append([a, b])
        out[m] = (L.rearrange(P), coefs)
    return out


def arm_arrays(W, months_P):
    return np.concatenate(months_P) if isinstance(months_P, list) else months_P


def threshold_block(y, P, t_idx_map, W=None):
    out = {}
    for t in EVAL_THRESHOLDS:
        k = list(L.ARR_THRESHOLDS).index(t)
        out[str(t)] = binary_metrics((y >= t).astype(float), P[:, k])
    return out


def arm_report(W, P, climo_rates):
    y = W["y"]
    k15 = list(L.ARR_THRESHOLDS).index(15)
    p15 = P[:, k15]
    y15 = (y >= 15).astype(float)
    rep = {"thresholds": threshold_block(y, P, None)}
    monthly = {}
    tiers_ok = True
    tier_rates = {}
    for ym in sorted(np.unique(W["ym"])):
        m = W["ym"] == ym
        e, _ = deciles(y15[m], p15[m])
        row = {"n": int(m.sum()), "ece": float(e), "auc": float(roc_auc_score(y15[m], p15[m]))}
        tr = tier_of(p15[m])
        rates = [float(y15[m][tr == i].mean()) if (tr == i).any() else None for i in range(3)]
        row["tier_observed"] = rates
        row["tier_ordered"] = ordered_rates(rates, strict=True)
        tiers_ok &= row["tier_ordered"]
        monthly[ym] = row
    rep["monthly_t15"] = monthly
    tr = tier_of(p15)
    rates = [float(y15[tr == i].mean()) if (tr == i).any() else None for i in range(3)]
    rep["tiers"] = {"observed": rates, "counts": [int((tr == i).sum()) for i in range(3)],
                    "high_over_low": (rates[2] / rates[0]) if rates[0] and rates[2] is not None else None,
                    "ordered_every_month": bool(tiers_ok)}
    # ranked probability score against the fit-window climatology ladder
    ind = (y[:, None] >= np.asarray(L.ARR_THRESHOLDS)[None, :]).astype(float)
    rps = ((P - ind) ** 2).sum(1)
    rps_c = ((np.asarray(climo_rates)[None, :] - ind) ** 2).sum(1)
    rep["rps"] = {"model": float(rps.mean()), "climatology": float(rps_c.mean()),
                  "skill": day_block_ci(W["day"], rps, rps_c, kind="skill")}
    rep["rearranged_share"] = float(W["rearr"].mean())
    rep["mean_abs_dep15_minus_production"] = float(np.abs(W["dep"][:, list(L.DEP_THRESHOLDS).index(15)]
                                                          - W["p_prod"]).mean())
    nk = W["carrier"] != "NK"
    rep["without_NK_t15"] = binary_metrics(y15[nk], p15[nk])
    return rep, rps, rps_c


def coverage(P, y, chunk=20000):
    """Share of outcomes inside [q10, q90], and below q50 (integer outcomes taken at the cell centre y + 0.5)."""
    inside = below = 0
    yc = np.where(np.isfinite(y), y + 0.5, np.inf)
    for lo in range(0, len(P), chunk):
        S = L.curve_matrix(P[lo:lo + chunk].astype(float), L.ARR_THRESHOLDS, TAIL_ARR)
        q10, q50, q90 = (L.quantile_matrix(S, q) for q in (0.1, 0.5, 0.9))
        yy = yc[lo:lo + chunk]
        inside += int(((yy >= q10) & (yy <= q90)).sum())
        below += int((yy <= q50).sum())
    return {"q10_q90": inside / len(P), "below_q50": below / len(P)}


TAIL_ARR = None   # set in run()


def arrival_evaluation(store, cfg, log):
    global TAIL_ARR
    TAIL_ARR = store.bundle.arr_tail
    val = months_between(*cfg["val"])
    W = collect_window(store, val)
    log(f"arrival rows {len(W['y'])}")
    # climatology ladder from the fit window (flown flights, 2025-01..06 labels only)
    fit_y = np.concatenate([store.meta(ym)["arr_delay"].to_numpy(float)[flown_arrival_rows(store.meta(ym))]
                            for ym in months_between(*FIT_MONTHS)])
    climo = [float((fit_y >= t).mean()) for t in L.ARR_THRESHOLDS]
    lr = recalibrated_arm(store, val)
    P_L = W["P"].astype(float)
    P_LR = np.concatenate([lr[m][0] for m in val])
    k15 = list(L.ARR_THRESHOLDS).index(15)
    y15 = (W["y"] >= 15).astype(float)
    sel = np.isin(W["ym"], months_between(*cfg["select"]))
    con = np.isin(W["ym"], months_between(*cfg["confirm"]))
    b = lambda P, m: float(np.mean((P[m, k15] - y15[m]) ** 2))        # noqa: E731
    sq = lambda P: (P[:, k15] - y15) ** 2                              # noqa: E731
    diff_con = day_block_ci(W["day"][con], (sq(P_LR) - sq(P_L))[con])
    lr_served = bool(b(P_LR, sel) < b(P_L, sel) and diff_con["hi"] < 0)
    arm_choice = {"brier_L_select": b(P_L, sel), "brier_LR_select": b(P_LR, sel),
                  "brier_diff_LR_minus_L_confirm": diff_con, "served_arm": "L-R" if lr_served else "L"}
    log(f"arm choice: {arm_choice['served_arm']}")
    reports, chosen = {}, None
    for name, P in (("L", P_L), ("L-R", P_LR)):
        rep, rps, rps_c = arm_report(W, P, climo)
        reports[name] = rep
        if name == arm_choice["served_arm"]:
            chosen = (name, P, rps, rps_c)
    name, P, rps, rps_c = chosen
    # skill and coverage for the chosen arm
    p15 = P[:, k15]
    base_prod = (W["p_prod"].astype(float) - y15) ** 2
    base_clim = (climo[k15] - y15) ** 2
    d_prod = day_block_ci(W["day"], sq(P) - base_prod)
    d_clim = day_block_ci(W["day"], sq(P) - base_clim)
    cov = coverage(P, W["y"])
    log(f"coverage {cov}")
    rep = reports[name]
    t = rep["thresholds"]
    A1 = {"ece_t15_le_0.02": t["15"]["ece"] <= 0.02,
          "max_decile_gap_t15_le_0.04": t["15"]["max_decile_gap"] <= 0.04,
          "monthly_ece_le_0.04": all(v["ece"] <= 0.04 for v in rep["monthly_t15"].values()),
          "ece_t45_le_0.03": t["45"]["ece"] <= 0.03, "ece_t120_le_0.03": t["120"]["ece"] <= 0.03}
    A2 = {"brier_below_production_dep_baseline_ci": d_prod, "brier_below_climatology_ci": d_clim,
          "pass_vs_production": d_prod["hi"] < 0, "pass_vs_climatology": d_clim["hi"] < 0}
    A3 = {"high_ge_1.8x_low": (rep["tiers"]["high_over_low"] or 0) >= 1.8,
          "ordered_every_month": rep["tiers"]["ordered_every_month"]}
    A4 = {"rps_skill": rep["rps"]["skill"], "rps_skill_positive_ci_excludes_0": rep["rps"]["skill"]["lo"] > 0,
          "coverage": cov, "coverage_within_0.80_pm_0.04": abs(cov["q10_q90"] - 0.80) <= 0.04}
    a1, a2, a3, a4 = (all(v for v in A1.values()), A2["pass_vs_production"] and A2["pass_vs_climatology"],
                      all(A3.values()), A4["rps_skill_positive_ci_excludes_0"] and A4["coverage_within_0.80_pm_0.04"])
    if a1 and a2 and a3:
        outcome = "ship_arrival_in_predict_and_headline"
    elif a1 and a4 and not a2:
        outcome = "ladder_inside_connection_engine_only"
    elif not a1:
        outcome = "ship_nothing_arrival"
    else:
        outcome = "ship_nothing_arrival_other"
    decisions = {"A1": {"pass": a1, "detail": A1}, "A2": {"pass": a2, "detail": A2}, "A3": {"pass": a3, "detail": A3},
                 "A4": {"pass": a4, "detail": A4}, "outcome": outcome}
    return {"climatology_ladder": dict(zip(map(str, L.ARR_THRESHOLDS), climo)), "arm_choice": arm_choice,
            "arms": reports, "served_arm": name, "decisions": decisions,
            "baselines_t15": {"production_dep_as_arrival": binary_metrics(y15, W["p_prod"].astype(float)),
                              "climatology_brier": float(np.mean(base_clim))}}


# ------------------------------------------------------------------ connections

def gather_pairs(store, pairs):
    """Per-pair ladder rows, cancellation probabilities and leg metadata for one pair-file frame."""
    cols = {k: [] for k in ("armP", "depP", "c1", "c2", "hour2", "same_carrier", "carrier1")}
    order = []
    for ym2 in pairs["ym2"].unique():
        sub = pairs[pairs["ym2"] == ym2]
        ym1 = sub["ym"].iloc[0]
        p1, p2 = store.pred(ym1), store.pred(ym2)
        m1, m2 = store.meta(ym1), store.meta(ym2)
        r1, r2 = sub["row1"].to_numpy(), sub["row2"].to_numpy()
        cols["armP"].append(p1["arr"][r1])
        cols["depP"].append(p2["dep"][r2])
        cols["c1"].append(p1["cancel"][r1])
        cols["c2"].append(p2["cancel"][r2])
        cols["hour2"].append(pd.to_datetime(m2["dep_wall"]).dt.hour.to_numpy()[r2])
        c1, c2 = m1["carrier"].to_numpy()[r1], m2["carrier"].to_numpy()[r2]
        cols["same_carrier"].append(c1 == c2)
        cols["carrier1"].append(c1)
        order.append(sub.index.to_numpy())
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["index"] = np.concatenate(order)
    return out


def load_pairs(ym, frac):
    p = pd.read_parquet(ad.CACHE_DIR / f"pairs_{ym}.parquet")
    if frac < 1.0:
        p = p.sample(frac=frac, random_state=SEED).sort_index()
    return p


def pit_sample(store, pairs, dep_model=None):
    """Normal scores of the PIT of X (leg 1 arrival delay) and Y (leg 2 departure delay) for pairs where both legs
    operated, using the ladder marginals; randomised within the integer bin (seed 42)."""
    g = gather_pairs(store, pairs)
    P = pairs.loc[g["index"]]
    ok = ~P["c1"].to_numpy() & ~P["c2"].to_numpy() & ~P["div1"].to_numpy() & np.isfinite(P["x"].to_numpy()) \
        & np.isfinite(P["y"].to_numpy())
    rng = np.random.default_rng(SEED)
    z1 = C.pit_normal_scores(g["armP"][ok].astype(float), L.ARR_THRESHOLDS, store.bundle.arr_tail,
                             P["x"].to_numpy()[ok], rng)
    z2 = C.pit_normal_scores(g["depP"][ok].astype(float), L.DEP_THRESHOLDS, store.bundle.dep_tail,
                             P["y"].to_numpy()[ok], rng)
    return z1, z2, P["hub"].to_numpy()[ok], P["layover"].to_numpy()[ok]


def score_month(store, ym, cfg, rho_models: dict, m_shift=0, subsample=None):
    """P(miss) for every proxy pair of month ym under each dependence model, plus labels and features."""
    pairs = load_pairs(ym, cfg["frac"])
    if subsample is not None:
        pairs = pairs.sample(frac=subsample, random_state=SEED)
    g = gather_pairs(store, pairs)
    P = pairs.loc[g["index"]]
    lay, M = P["layover"].to_numpy().astype(int), P["M"].to_numpy().astype(int) + m_shift
    rhos = {k: dm.rho_vector(P["hub"].to_numpy(), lay) for k, dm in rho_models.items()}
    p = C.miss_batch(g["armP"], g["depP"], store.bundle.arr_tail, store.bundle.dep_tail, lay - M, rhos)
    miss, disr = pr.outcomes(P["x"].to_numpy(float), P["y"].to_numpy(float), P["c1"].to_numpy(bool),
                             P["c2"].to_numpy(bool), P["div1"].to_numpy(bool), lay, M)
    th_a, th_d = list(L.ARR_THRESHOLDS), list(L.DEP_THRESHOLDS)
    arr_curve_s1 = C.sf_at(g["armP"].astype(float), L.ARR_THRESHOLDS, store.bundle.arr_tail, (lay - M).astype(float))
    dep_s0 = g["depP"][:, th_d.index(0)].astype(float)
    return {"p": {k: v.astype(np.float32) for k, v in p.items()}, "miss": miss, "disrupted": disr,
            "c1p": g["c1"], "c2p": g["c2"], "layover": lay, "M": M, "bucket": P["bucket"].to_numpy(),
            "w": P["w"].to_numpy(float), "hub": P["hub"].to_numpy(), "network": P["network"].to_numpy(),
            "ym": np.full(len(P), ym), "day": P["arr_date"].to_numpy().astype("datetime64[D]").astype(int),
            "same_carrier": g["same_carrier"], "hour2": g["hour2"], "s1": arr_curve_s1, "s2_0": dep_s0}


def cat(blocks):
    return {k: (np.concatenate([b[k] for b in blocks]) if k != "p" else
                {n: np.concatenate([b["p"][n] for b in blocks]) for n in blocks[0]["p"]}) for k in blocks[0]}


def layover_group(layover):
    return np.where(layover < 60, "lt60", np.where(layover < 120, "60_120", "120_240"))


def connection_evaluation(store, cfg, log):
    rho_fit = months_between(*cfg["rho_fit"])
    rho_sel = months_between(*cfg["rho_sel"])
    val = months_between(*cfg["val"])
    assert_window(rho_fit + rho_sel + val)
    # 1. rho from 2025-07..09 pairs
    zs = [pit_sample(store, load_pairs(ym, cfg["frac"])) for ym in rho_fit]
    z1, z2, hubs, lay = (np.concatenate(x) for x in zip(*zs))
    dm = {v: C.fit_dependence(z1, z2, hubs, lay, variant=v) for v in ("R0", "R1", "R2")}
    log(f"rho fit on {len(z1)} pairs: global {dm['R1'].global_rho:.3f}")
    # 2. score fit and selection months under all three variants; isotonic per variant on the fit months
    from sklearn.isotonic import IsotonicRegression
    fit_blocks = cat([score_month(store, ym, cfg, dm) for ym in rho_fit])
    sel_blocks = cat([score_month(store, ym, cfg, dm) for ym in rho_sel])
    isos = {v: IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(fit_blocks["p"][v], fit_blocks["miss"].astype(float))
            for v in dm}
    sel_scores = {}
    for v in dm:
        for recal in (False, True):
            p = isos[v].predict(sel_blocks["p"][v]) if recal else sel_blocks["p"][v]
            sel_scores[v + ("+iso" if recal else "")] = float(brier_score_loss(sel_blocks["miss"], p))
    chosen = min(sel_scores, key=sel_scores.get)
    variant, recal = chosen.split("+")[0], chosen.endswith("+iso")
    log(f"variant selection (Brier on {rho_sel[0]}..{rho_sel[-1]}): {sel_scores} -> {chosen}")
    # 3. refit the chosen variant on all of 2025-07..12
    if variant != "R0":
        zs += [pit_sample(store, load_pairs(ym, cfg["frac"])) for ym in rho_sel]
        z1, z2, hubs, lay = (np.concatenate(x) for x in zip(*zs))
        final_dm = C.fit_dependence(z1, z2, hubs, lay, variant=variant)
    else:
        final_dm = dm["R0"]
    final_iso = None
    if recal:
        if variant == "R0":
            allb = cat([fit_blocks, sel_blocks])
            final_iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(allb["p"]["R0"], allb["miss"].astype(float))
        else:
            allb = cat([score_month(store, ym, cfg, {variant: final_dm}) for ym in rho_fit + rho_sel])
            final_iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(allb["p"][variant], allb["miss"].astype(float))
    base_blocks = cat([fit_blocks, sel_blocks])
    # baseline B-L from 2025-07..12 pairs (unweighted)
    bl_tab = base_rate_table(base_blocks)
    # diagnostic classifier D on 2025-07..09 pairs
    diag = diagnostic_classifier(fit_blocks)
    del fit_blocks, sel_blocks, base_blocks
    # 4. evaluation months
    models = {variant: final_dm, "R0": dm["R0"]}
    ev = cat([score_month(store, ym, cfg, models) for ym in val])
    pm = final_iso.predict(ev["p"][variant]) if final_iso is not None else ev["p"][variant].astype(float)
    pm = np.asarray(pm, dtype=float)
    p_i = ev["p"]["R0"].astype(float)
    p_bl = bl_lookup(bl_tab, ev["hub"], ev["layover"])
    res = connection_metrics(ev, pm, p_i, p_bl)
    res["variant_selection_brier"] = sel_scores
    res["chosen_variant"] = chosen
    res["rho"] = {"variant": variant, "global": final_dm.global_rho, "cells": len(final_dm.by_cell),
                  "fit_n_2025_07_09": int(len(z1))}
    res["diagnostic_classifier_D"] = diag_eval(diag, ev, pm)
    res["sensitivity_mct_pm10"] = sensitivity(store, cfg, val, variant, final_dm, final_iso)
    res["_final_dm"] = {"variant": variant, "global_rho": final_dm.global_rho,
                        "by_cell": {k: list(v) for k, v in final_dm.by_cell.items()}, "shrink_n": final_dm.shrink_n,
                        "recalibration": None if final_iso is None else
                        {"type": "isotonic", "x": final_iso.X_thresholds_.tolist(), "y": final_iso.y_thresholds_.tolist()},
                        "chosen": chosen, "decision_outcome": res["decisions"]["outcome"],
                        "protocol": "backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md"}
    return res


def base_rate_table(b):
    df = pd.DataFrame({"hub": b["hub"], "cell": b["layover"] // 5, "miss": b["miss"].astype(float),
                       "g": layover_group(b["layover"])})
    return {"cell": df.groupby(["hub", "cell"])["miss"].mean().to_dict(),
            "bucket": df.groupby(["g", "cell"])["miss"].mean().to_dict(), "all": float(df["miss"].mean())}


def bl_lookup(tab, hubs, layover):
    out = np.empty(len(hubs))
    cell = layover // 5
    g = layover_group(layover)
    for i, (h, c, gg) in enumerate(zip(hubs, cell, g)):
        v = tab["cell"].get((h, c))
        if v is None:
            v = tab["bucket"].get((gg, c), tab["all"])
        out[i] = v
    return out


def diag_features(b):
    th_hubs = {h: i for i, h in enumerate(mctmod.TOP30)}
    nets = {n: i for i, n in enumerate(sorted(["AA", "AS", "B6", "DL", "F9", "G4", "UA", "WN", "YX", "OO", "OH", "MQ", "9E", "QX", "EV", "YV", "NK", "HA", "SY", "XP"]))}
    return np.column_stack([b["layover"], b["M"], [th_hubs.get(h, -1) for h in b["hub"]],
                            [nets.get(n, -1) for n in b["network"]], b["hour2"],
                            pd.Series(np.asarray(b["day"]).astype("datetime64[D]")).dt.month.to_numpy(),
                            b["s1"], b["s2_0"]]).astype(np.float32)


def diagnostic_classifier(b):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=SEED)
    m.fit(diag_features(b), b["miss"].astype(int))
    return m


def diag_eval(model, ev, pm):
    pD = model.predict_proba(diag_features(ev))[:, 1]
    y = ev["miss"].astype(float)
    d = binary_metrics(y, pD)
    d.pop("deciles", None)
    return {"metrics": d, "brier_structural": float(brier_score_loss(y, pm)),
            "note": "diagnostic only; never shipped from this protocol"}


def group_table(y, p, labels, min_n=1):
    out = {}
    for g in pd.unique(labels):
        m = labels == g
        if m.sum() >= min_n:
            out[str(g)] = {"n": int(m.sum()), "mean_pred": float(p[m].mean()), "observed": float(y[m].mean()),
                           "gap": float(p[m].mean() - y[m].mean()), "brier": float(np.mean((p[m] - y[m]) ** 2))}
    return out


def connection_metrics(ev, pm, p_i, p_bl):
    y = ev["miss"].astype(float)
    w = ev["w"]
    res = {"n_pairs": int(len(y)), "observed_miss_rate": float(y.mean()),
           "weighted_observed_miss_rate": float(np.average(y, weights=w))}
    res["model"] = binary_metrics(y, pm)
    res["weighted"] = {"brier": float(np.average((pm - y) ** 2, weights=w)), "ece": weighted_ece(y, pm, w),
                       "mean_pred": float(np.average(pm, weights=w)), "observed": float(np.average(y, weights=w))}
    res["baselines"] = {"B-L": binary_metrics(y, p_bl, full=False), "B-I_R0": binary_metrics(y, p_i, full=False)}
    lg = layover_group(ev["layover"])
    res["by_layover_bucket"] = group_table(y, pm, lg)
    res["by_sampling_bucket"] = group_table(y, pm, ev["bucket"])
    top10 = pd.Series(ev["hub"]).value_counts().index[:10]
    sel = np.isin(ev["hub"], top10)
    res["by_hub_top10"] = group_table(y[sel], pm[sel], ev["hub"][sel])
    res["by_month"] = {}
    months_ok = True
    c6_ok = True
    for ym in sorted(np.unique(ev["ym"])):
        m = ev["ym"] == ym
        e, _ = deciles(y[m], pm[m])
        bnd = band_of(pm[m])
        rates = [float(y[m][bnd == i].mean()) if (bnd == i).any() else None for i in range(3)]
        ordered = ordered_rates(rates)
        c6_ok &= ordered
        res["by_month"][ym] = {"n": int(m.sum()), "ece": float(e), "band_observed": rates, "bands_ordered": ordered,
                               "mean_pred": float(pm[m].mean()), "observed": float(y[m].mean())}
        months_ok &= e <= 0.03
    res["by_same_carrier"] = group_table(y, pm, np.where(ev["same_carrier"], "same_operating_carrier", "different"))
    # trip level
    pdis = 1 - (1 - ev["c1p"].astype(float)) * (1 - ev["c2p"].astype(float)) * (1 - pm)
    yd = ev["disrupted"].astype(float)
    res["disrupted"] = binary_metrics(yd, pdis)
    # decisions
    mm = res["model"]
    c3 = {k: abs(v["gap"]) <= 0.02 for k, v in res["by_layover_bucket"].items() if v["n"] >= 10000}
    d_bl = day_block_ci(ev["day"], (pm - y) ** 2 - (p_bl - y) ** 2)
    C1 = {"ece_le_0.015": mm["ece"] <= 0.015, "max_gap_deciles_1_9_le_0.03": mm["max_gap_deciles_1_9"] <= 0.03,
          "top_decile_gap_le_0.05": mm["top_decile_gap"] <= 0.05}
    dec = {"C1": {"pass": all(C1.values()), "detail": C1},
           "C2": {"pass": bool(months_ok), "detail": {k: v["ece"] for k, v in res["by_month"].items()}},
           "C3": {"pass": all(c3.values()), "detail": c3},
           "C4": {"pass": bool(d_bl["hi"] < 0), "detail": {"brier_diff_vs_B-L": d_bl}},
           "C5": {"pass": res["disrupted"]["ece"] <= 0.02, "detail": {"ece": res["disrupted"]["ece"]}},
           "C6": {"pass": bool(c6_ok)}}
    c1, c2, c3p, c4, c6 = (dec[k]["pass"] for k in ("C1", "C2", "C3", "C4", "C6"))
    if c1 and c2 and c3p and c4 and c6:
        outcome = "ship_miss_probability_for_validated_connections"
    elif c1 and c2 and c3p and not c4:
        outcome = "serve_layover_table_B-L_itin-layover-table-v1"
    elif not (c1 and c2 and c3p):
        outcome = "no_connection_numbers"
    else:
        outcome = "no_connection_numbers_c6_failed"
    dec["outcome"] = outcome
    dec["trip_level_number_shown"] = bool(dec["C5"]["pass"])
    res["decisions"] = dec
    return res


def sensitivity(store, cfg, val, variant, final_dm, final_iso):
    out = {}
    for d in (-10, 10):
        b = cat([score_month(store, ym, cfg, {variant: final_dm}, m_shift=d, subsample=0.10) for ym in val])
        p = final_iso.predict(b["p"][variant]) if final_iso is not None else b["p"][variant].astype(float)
        m = binary_metrics(b["miss"].astype(float), np.asarray(p, dtype=float))
        m.pop("deciles", None)
        out[f"M{d:+d}"] = m
    return out


# ------------------------------------------------------------------ report

def render_report(R):
    a, c = R["arrival"], R["connections"]
    L_ = ["# Arrival delay and missed connections: evaluation report (draft tables)", "",
          f"Protocol: `backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md`. Run on {R['meta']['run_date']}, artifact sha256 "
          f"`{R['meta']['artifact_sha256']}`. Evaluation months {R['meta']['val_months']}. This file is generated "
          "from `results/arrival_connections.json`; prose is added at review.", "",
          "## Arrival", f"Served arm: **{a['served_arm']}**. Outcome: **{a['decisions']['outcome']}**.", ""]
    for r in ("A1", "A2", "A3", "A4"):
        L_.append(f"- {r}: {'PASS' if a['decisions'][r]['pass'] else 'FAIL'}")
    L_ += ["", "| arm | t | AUC | Brier | ECE | max decile gap |", "|---|---|---|---|---|---|"]
    for arm, rep in a["arms"].items():
        for t, m in rep["thresholds"].items():
            L_.append(f"| {arm} | {t} | {m.get('auc', float('nan')):.3f} | {m['brier']:.4f} | {m['ece']:.4f} | "
                      f"{m['max_decile_gap']:.4f} |")
    L_ += ["", "## Connections", f"Variant: **{c['chosen_variant']}**. Outcome: **{c['decisions']['outcome']}**.", ""]
    for r in ("C1", "C2", "C3", "C4", "C5", "C6"):
        L_.append(f"- {r}: {'PASS' if c['decisions'][r]['pass'] else 'FAIL'}")
    m = c["model"]
    L_ += ["", f"Pairs {c['n_pairs']:,}; observed miss rate {c['observed_miss_rate']:.4f} (reweighted "
           f"{c['weighted_observed_miss_rate']:.4f}). Brier {m['brier']:.4f}, ECE {m['ece']:.4f}, AUC {m.get('auc', float('nan')):.3f}. "
           f"B-L Brier {c['baselines']['B-L']['brier']:.4f}.", ""]
    return "\n".join(L_) + "\n"


# ------------------------------------------------------------------ main

def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def run(cfg, log=print):
    if ad.UNLOCK_VAR in os.environ:
        sys.exit(f"{ad.UNLOCK_VAR} is set; the evaluation never runs with it")
    all_months = set()
    for k in ("val", "rho_fit", "rho_sel"):
        all_months |= set(months_between(*cfg[k]))
    assert_window(sorted(all_months))
    bundle = L.load_bundle(ARTIFACT)
    sys.path.insert(0, str(ROOT))
    import main as app
    prod = app.model_service.model
    names = list(bundle.feature_names)
    store = Store(bundle, names, prod, sha256_file(ARTIFACT))
    t0 = time.time()
    arrival = arrival_evaluation(store, cfg, log)
    log(f"arrival done {time.time() - t0:.0f}s")
    conn = connection_evaluation(store, cfg, log)
    log(f"connections done {time.time() - t0:.0f}s")
    return {"meta": {"protocol": "backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md", "mode": cfg["name"],
                     "run_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
                     "artifact_sha256": store.sha, "val_months": [cfg["val"][0], cfg["val"][1]],
                     "seconds": round(time.time() - t0)},
            "arrival": arrival, "connections": conn}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="2025 months only; smoke test, never a result")
    args = ap.parse_args(argv)
    cfg = QUICK if args.quick else FULL
    R = run(cfg, log=lambda s: print(s, flush=True))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / cfg["out"]).write_text(json.dumps(jsonable(R), indent=1) + "\n")
    if not args.quick:
        # connection_dependence_v1.json: {variant, global_rho, by_cell: {"HUB|bucket": [rho, n]}, shrink_n} plus
        # recalibration/decision fields the API loader ignores. Written only by the full run.
        (ROOT / "models" / "connection_dependence_v1.json").write_text(
            json.dumps(jsonable(R["connections"]["_final_dm"]), indent=1) + "\n")
        (Path(__file__).resolve().parent / "REPORT_ARRIVAL_CONNECTIONS.md").write_text(render_report(R))
    print("arrival outcome:", R["arrival"]["decisions"]["outcome"])
    print("connection outcome:", R["connections"]["decisions"]["outcome"])


if __name__ == "__main__":
    main()
