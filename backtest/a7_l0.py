"""
A7 + L0: value of inbound-aircraft linkage, and stability of aircraft rotations.  Implements backtest/PROTOCOL_A7_L0.md
(pre-registered, committed before any evaluation).

    python backtest/a7_l0.py            # writes backtest/results/a7_l0.json
    python backtest/a7_l0.py --quick    # tiny debug run on 2025 months (writes a7_l0_quick.json)

Validation months only (2026-01..08 evaluation); nothing from 2026-09 on is loaded.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import backtest as bt  # noqa: E402
import runup  # noqa: E402
from flyrely_features._common import HORIZONS, cutoff_day  # noqa: E402
from inbound import IN_COLS, Pairings, inbound_features, plurality_prev_weeks  # noqa: E402

QUICK = runup.QUICK
CACHE = HERE.parent / "data" / "cache" / ("a7_l0_quick" if QUICK else "a7_l0")
RESULTS = HERE / "results"
FORBIDDEN_FROM = pd.Timestamp("2026-09-01")
HORIZON_LIST = ["3h", "1h"]
SEED = 42
LEVELS = [0.25, 0.29, 0.5, 0.7, 0.75, 0.9]
HUBS = ["DEN", "LGA", "SEA", "DFW", "ORD", "CLT", "DTW", "BNA", "ATL", "MSP", "MDW", "RDU"]
# decision thresholds (PROTOCOL_A7_L0.md)
WORTH = 0.0005
STABLE = 0.60


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def link_mask(u: np.ndarray, p: float, regional: np.ndarray = None) -> np.ndarray:
    """Flights whose aircraft is 'linked'. u is a fixed uniform per flight, so linked sets are nested as p grows.
    With `regional`, regional flights are never linked and mainline flights are linked with probability p."""
    linked = u < p
    return linked if regional is None else linked & ~regional


def mask_columns(X: pd.DataFrame, cols: list, linked: np.ndarray) -> pd.DataFrame:
    X = X.copy()
    X.loc[~linked, cols] = np.nan
    return X


def decide(res: dict) -> dict:
    """Pre-registered decision rules 1-4 from the T-3h results (dict shaped like the JSON `3h` section + `l0`)."""
    g = res["3h"]["gain_vs_N"]
    out = {}
    perfect = g["P+"]
    if perfect["diff"] < WORTH:
        out["linkage_priority"] = "no: perfect linkage worth < 0.0005 Brier at T-3h; skip AeroDataBox"
    elif perfect["ci95"][0] < WORTH:
        out["linkage_priority"] = "inconclusive, lean skip"
    else:
        out["linkage_priority"] = "yes: worth pursuing"
    g29 = g["L0.29"]
    out["sfdps_level_useful"] = bool(g29["ci95"][0] > 0 and g29["diff"] >= WORTH)
    gap = res["3h"]["perfect_minus_level"]
    out["paid_justified_vs_0.5"] = bool(gap["0.5"]["gain_of_perfect"] >= WORTH and gap["0.5"]["ci95"][0] > 0)
    pr = g["PR"]
    l0 = res["l0"]["overall"]["d-7"]["p_match_given_exists"]
    out["prior_worth_building"] = bool(l0 >= STABLE and pr["ci95"][0] > 0 and pr["diff"] >= WORTH)
    out["prior_strong"] = bool(pr["diff"] >= 0.5 * perfect["diff"])
    return out


# =============================================================================
# L0: pairing stability
# =============================================================================

def rate(match: np.ndarray, exists: np.ndarray, have: np.ndarray) -> dict:
    """have = flights with a pairing today; exists = a pairing existed on the earlier day; match = same inbound flight."""
    n_have, n_exists = int(have.sum()), int((have & exists).sum())
    n_match = int((have & exists & match).sum())
    return {"n": n_have, "coverage": round(n_exists / max(n_have, 1), 4),
            "p_match_given_exists": round(n_match / max(n_exists, 1), 4),
            "hit_over_all": round(n_match / max(n_have, 1), 4)}


def l0_stability(A, df, P: Pairings, ev_rows: np.ndarray) -> dict:
    """Stability of today's inbound->outbound pairing versus d-1, d-7 and the 4-week plurality, on rows `ev_rows`."""
    rows = ev_rows
    have = P.prev[rows] >= 0
    today = np.where(have, A["fn_id"][np.where(have, P.prev[rows], 0)], -1)
    day = A["day"][rows]
    pred = {}
    for name, lag in (("d-1", 1), ("d-7", 7)):
        f, _ = P.fn_a_on(rows, day - lag)
        pred[name] = f
    pred["plurality_4w"] = plurality_prev_weeks(P, rows, 4)
    regional = (df["Reporting_Airline"].to_numpy() != df["Marketing_Airline_Network"].to_numpy())[rows]
    origin = A["airports"][A["o_id"][rows]].to_numpy()
    carrier = df["Reporting_Airline"].to_numpy()[rows]
    wkend = df["dow"].to_numpy()[rows] >= 6
    through = np.where(have, today == A["fn_id"][rows], False)

    def block(sel):
        return {k: rate(pred[k] == today, pred[k] >= 0, have & sel) for k in pred}

    everything = np.ones(len(rows), dtype=bool)
    out = {"overall": block(everything), "through_flight_share": round(float(through[have].mean()), 4),
           "n_flights_with_pairing": int(have.sum()), "n_flights": int(len(rows)),
           "mainline": block(~regional), "regional": block(regional), "weekday": block(~wkend), "weekend": block(wkend),
           "hubs": {h: block(origin == h) for h in HUBS},
           "by_ym": {ym: block((df["ym"].to_numpy()[rows] == ym)) for ym in sorted(set(df["ym"].to_numpy()[rows]))}}
    top = pd.Series(carrier[have]).value_counts().head(12).index
    out["carriers"] = {c: block(carrier == c) for c in top}
    return out


# =============================================================================
# A7
# =============================================================================

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    df, A = runup.load()
    if df["FlightDate"].max() >= FORBIDDEN_FROM:
        raise RuntimeError("protected holdout data (>= 2026-09) present")
    W = runup.Weather(A["airports"])
    C, R = runup.build_counters(A), runup.Rotation(A)
    P = Pairings(A, R)
    PERIODS = {"fit": runup.PERIODS["fit"], "val": runup.PERIODS["val"], "eval": runup.PERIODS["holdout"]}
    within = lambda a, b: np.flatnonzero(((df["ym"] >= a) & (df["ym"] <= b)).to_numpy())
    rows = {k: within(*v["rows"]) for k, v in PERIODS.items()}
    frame = {k: df.iloc[r] for k, r in rows.items()}
    booking_X = {k: bt.booking_features(frame[k], df.iloc[within(*v["rates"])]).astype("float32") for k, v in PERIODS.items()}
    flown = {k: ~frame[k]["cancelled"].to_numpy() for k in rows}
    y = {k: frame[k]["delayed"].to_numpy()[flown[k]] for k in rows}
    ev = frame["eval"]
    U = np.random.default_rng(SEED).random(len(df))
    regional_all = df["Reporting_Airline"].to_numpy() != df["Marketing_Airline_Network"].to_numpy()

    results = {"protocol": "backtest/PROTOCOL_A7_L0.md", "quick_debug_run": QUICK, "periods": PERIODS,
              "levels": LEVELS, "eval_flights": int(len(ev)), "eval_flown": int(flown["eval"].sum())}

    results["l0"] = l0_stability(A, df, P, rows["eval"])
    log("L0 overall", results["l0"]["overall"])

    for h in HORIZON_LIST:
        X, prior_hit = {}, {}
        for k in ("fit", "val", "eval"):
            f = CACHE / f"X_{h}_{k}.parquet"
            c = A["s_dep"][rows[k]] - HORIZONS[h]
            prev_pr = P.prior_inbound(rows[k], cutoff_day(A, rows[k], h) - 1)
            tail_ok = (prev_pr >= 0) & (A["tail_id"][np.where(prev_pr >= 0, prev_pr, 0)] == A["tail_id"][rows[k]]) & (A["tail_id"][rows[k]] >= 0)
            prior_hit[k] = {"coverage": float((prev_pr >= 0).mean()), "precision": float(tail_ok.sum() / max((prev_pr >= 0).sum(), 1)),
                            "hit_over_all": float(tail_ok.mean())}
            if f.exists():
                X[k] = pd.read_parquet(f)
                continue
            base = runup.horizon_features(A, C, R, W, rows[k], h, booking_X[k], W.ok)
            true_in = inbound_features(A, rows[k], P.prev[rows[k]], c)
            prior_in = inbound_features(A, rows[k], prev_pr, c)
            extra = {**{c_: v for c_, v in true_in.items()}, **{"pr_" + c_: v for c_, v in prior_in.items()}}
            X[k] = pd.concat([base, pd.DataFrame(extra, index=base.index).astype("float32")], axis=1)
            X[k].to_parquet(f)
        rot_cols = [c for c in X["fit"].columns if c.startswith("rot_")]
        in_cols = [c for c in X["fit"].columns if c.startswith("in_")]
        pr_cols = [c for c in X["fit"].columns if c.startswith("pr_in_")]
        base_cols = [c for c in X["fit"].columns if c not in rot_cols + in_cols + pr_cols]
        link_cols = rot_cols + in_cols
        assert set(IN_COLS) == set(in_cols), in_cols
        log(h, "features", len(base_cols), "base", len(rot_cols), "rot", len(in_cols), "in", len(pr_cols), "prior")

        def arm_frame(k, cols, linked=None):
            Z = X[k][cols]
            return Z if linked is None else mask_columns(Z, [c for c in cols if c in link_cols], linked)

        def fit_arm(name, cols, linked_fn=None):
            pf = CACHE / f"pred_{h}_{name}.npy"
            mf = CACHE / f"model_{h}_{name}.joblib"
            if pf.exists():
                return np.load(pf), (joblib.load(mf) if mf.exists() else None)
            lk = {k: (linked_fn(rows[k]) if linked_fn else None) for k in ("fit", "val", "eval")}
            Xf, Xv, Xe = (arm_frame(k, cols, lk[k]) for k in ("fit", "val", "eval"))
            log(h, "fitting", name, len(cols), "features")
            m, iso = bt.fit_calibrated(Xf.loc[flown["fit"]], y["fit"], Xv.loc[flown["val"]], y["val"])
            p = iso.predict(m.predict_proba(Xe)[:, 1])
            np.save(pf, p)
            joblib.dump((m, iso), mf)
            return p, (m, iso)

        preds, models = {}, {}
        specs = {"N": (base_cols, None), "P": (base_cols + rot_cols, None), "P+": (base_cols + link_cols, None)}
        for lv in LEVELS:
            specs[f"L{lv}"] = (base_cols + link_cols, (lambda lv: lambda r: link_mask(U[r], lv))(lv))
        specs["RG0.29"] = (base_cols + link_cols, lambda r: link_mask(U[r], 0.29, regional_all[r]))
        specs["RG1.0"] = (base_cols + link_cols, lambda r: link_mask(U[r], 1.0, regional_all[r]))
        for name, (cols, fn) in specs.items():
            preds[name], models[name] = fit_arm(name, cols, fn)
        # prior arm: base + predicted-inbound state under the standard names, no tail anywhere
        pf = CACHE / f"pred_{h}_PR.npy"
        if pf.exists():
            preds["PR"] = np.load(pf)
        else:
            ren = {c: c[3:] for c in pr_cols}
            Xf, Xv, Xe = (X[k][base_cols + pr_cols].rename(columns=ren) for k in ("fit", "val", "eval"))
            log(h, "fitting PR")
            m, iso = bt.fit_calibrated(Xf.loc[flown["fit"]], y["fit"], Xv.loc[flown["val"]], y["val"])
            preds["PR"] = iso.predict(m.predict_proba(Xe)[:, 1])
            np.save(pf, preds["PR"])
        # mismatch: the perfect-linkage model P+ served on degraded eval data
        if models["P+"] is None:
            raise SystemExit("P+ prediction cached but model missing: delete the cache files for this horizon and re-run")
        mP, isoP = models["P+"]
        for lv in LEVELS:
            pf = CACHE / f"pred_{h}_M{lv}.npy"
            if pf.exists():
                preds[f"M{lv}"] = np.load(pf)
                continue
            Xe = arm_frame("eval", base_cols + link_cols, link_mask(U[rows["eval"]], lv))
            preds[f"M{lv}"] = isoP.predict(mP.predict_proba(Xe)[:, 1])
            np.save(pf, preds[f"M{lv}"])

        r = {"n_features": {"base": len(base_cols), "rot": len(rot_cols), "inbound": len(in_cols)},
             "true_inbound_coverage_eval": round(float((P.prev[rows["eval"]] >= 0).mean()), 4),
             "prior_inbound": {k: {a: round(b, 4) for a, b in v.items()} for k, v in prior_hit.items()},
             "linked_share_eval": {f"L{lv}": round(float(link_mask(U[rows["eval"]], lv).mean()), 4) for lv in LEVELS}}
        r["linked_share_eval"]["RG0.29"] = round(float(link_mask(U[rows["eval"]], 0.29, regional_all[rows["eval"]]).mean()), 4)
        r["linked_share_eval"]["RG1.0"] = round(float(link_mask(U[rows["eval"]], 1.0, regional_all[rows["eval"]]).mean()), 4)
        r["regional_share_eval"] = round(float(regional_all[rows["eval"]].mean()), 4)
        r["arms"] = {a: runup.score(ev, p) for a, p in preds.items()}
        for a in r["arms"]:
            r["arms"][a].pop("deciles", None)
        r["gain_vs_N"] = {}
        for a in preds:
            if a == "N":
                continue
            d = runup.brier_diff(ev, preds["N"], preds[a])    # Brier(N) - Brier(arm): positive = arm better
            d["auc_change"] = round(r["arms"][a]["auc_delayed"] - r["arms"]["N"]["auc_delayed"], 4)
            d["ece_change"] = round(r["arms"][a]["ece"] - r["arms"]["N"]["ece"], 4)
            r["gain_vs_N"][a] = d
        # value of the remaining gap: Brier(level) - Brier(perfect P+), serve-trained
        r["perfect_minus_level"] = {}
        for lv in (0.29, 0.5, 0.75):
            d = runup.brier_diff(ev, preds[f"L{lv}"], preds["P+"])
            r["perfect_minus_level"][str(lv)] = {"gain_of_perfect": d["diff"], "ci95": d["ci95"]}
        r["mismatch_penalty"] = {}
        for lv in LEVELS:
            d = runup.brier_diff(ev, preds[f"M{lv}"], preds[f"L{lv}"])   # Brier(mismatch) - Brier(serve-trained)
            r["mismatch_penalty"][str(lv)] = {"diff": d["diff"], "ci95": d["ci95"]}
        results[h] = r
        log(h, {a: (r["arms"][a]["brier_delayed"], r["gain_vs_N"].get(a, {}).get("diff")) for a in r["arms"]})

    results["decisions"] = decide(results)
    log("decisions", results["decisions"])
    out = RESULTS / ("a7_l0_quick.json" if QUICK else "a7_l0.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    log("wrote", out)


if __name__ == "__main__":
    main()
