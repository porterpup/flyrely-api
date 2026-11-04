"""
B5: do observed METAR (A) and FAA GS/GDP programs (B) improve run-up delay prediction?  Implements
backtest/PROTOCOL_B5.md (pre-registered and committed before this script was run).

    python backtest/experiments_b5.py            # writes backtest/results/b5_metar_faa.json
    python backtest/experiments_b5.py --quick    # tiny debug run (2025 months only), writes b5_metar_faa_quick.json

Evaluation months 2026-01..08 are validation (PROTOCOL_ADDENDUM_2026-11.md); months >= 2026-09 are never loaded
(runup.load stops at 2026-08 and this script hard-fails on anything later).

Arms per horizon: base (existing run-up horizon model incl. archived-forecast weather), +A, +B, +A+B.
Decision rule per source and horizon: 95% day-block bootstrap CI of Brier(with) - Brier(base) entirely below 0 AND
ECE(with) <= ECE(base) + 0.005.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import backtest as bt  # noqa: E402
import runup  # noqa: E402
from fetch_faa_adv import OUT as FAA_DIR  # noqa: E402
from fetch_metar import busiest_airports  # noqa: E402
from flyrely_features import faa, metar  # noqa: E402
from flyrely_features._common import HORIZONS  # noqa: E402

QUICK = runup.QUICK
CACHE = HERE.parent / "data" / "cache" / ("b5_quick" if QUICK else "b5")
RESULTS = HERE / "results"
ECE_TOLERANCE = 0.005
FORBIDDEN_FROM = pd.Timestamp("2026-09-01")
# horizon -> sources tested (PROTOCOL_B5.md, "Horizons and availability")
TESTS = {"1d": ["B"], "3h": ["A", "B"], "1h": ["A", "B"]}
ARMS = {"base": "", "A": "A", "B": "B", "AB": "AB"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def decide(diff: dict, ece_with: float, ece_base: float, tol: float = ECE_TOLERANCE) -> dict:
    """Pre-registered rule: CI entirely below 0, and ECE not worse than base by more than `tol`."""
    ci_below_zero = diff["ci95"][1] < 0
    ece_ok = ece_with <= ece_base + tol
    return {"brier_ci_below_zero": bool(ci_below_zero), "ece_not_worse": bool(ece_ok), "adopt": bool(ci_below_zero and ece_ok)}


def monthly_brier_diff(frame: pd.DataFrame, p_a: np.ndarray, p_b: np.ndarray) -> dict:
    """Brier(a) - Brier(b) per month, flown flights only."""
    flown = ~frame["cancelled"].to_numpy()
    y = frame["delayed"].to_numpy()
    out = {}
    for ym, idx in frame.reset_index(drop=True).groupby("ym").indices.items():
        i = idx[flown[idx]]
        out[ym] = round(float(np.mean((p_a[i] - y[i]) ** 2) - np.mean((p_b[i] - y[i]) ** 2)), 6)
    return out


def load_programs() -> pd.DataFrame:
    f = FAA_DIR / "programs.parquet"
    if not f.exists():
        raise SystemExit(f"{f} missing: run backtest/fetch_faa_adv.py lists, details, build")
    p = pd.read_parquet(f)
    if p["sent_utc"].max() >= FORBIDDEN_FROM:
        raise RuntimeError("FAA programs contain advisories >= 2026-09 (protected holdout)")
    return p


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    df, A = runup.load()
    if df["FlightDate"].max() >= FORBIDDEN_FROM:
        raise RuntimeError("protected holdout data (>= 2026-09) present")
    W = runup.Weather(A["airports"])
    C, R = runup.build_counters(A), runup.Rotation(A)
    M = metar.Metar(A["airports"])
    prog = load_programs()
    covered = busiest_airports(100)
    P = faa.Programs(prog, A["airports"], covered)
    log(f"METAR airports {M.n_airports if M.ok else 0}; FAA airports {P.n_airports}; advisories {len(prog)}; "
        f"missing-period GS/GDP {P.chains['GS'].n_missing_period}/{P.chains['GDP'].n_missing_period}")

    PERIODS = {"fit": runup.PERIODS["fit"], "val": runup.PERIODS["val"], "eval": runup.PERIODS["holdout"]}
    within = lambda a, b: np.flatnonzero(((df["ym"] >= a) & (df["ym"] <= b)).to_numpy())
    rows = {k: within(*v["rows"]) for k, v in PERIODS.items()}
    frame = {k: df.iloc[r] for k, r in rows.items()}
    booking_X = {k: bt.booking_features(frame[k], df.iloc[within(*v["rates"])]).astype("float32") for k, v in PERIODS.items()}
    flown = {k: ~frame[k]["cancelled"].to_numpy() for k in rows}
    y = {k: frame[k]["delayed"].to_numpy()[flown[k]] for k in rows}
    ev = frame["eval"]

    results = {"protocol": "backtest/PROTOCOL_B5.md", "quick_debug_run": QUICK, "periods": PERIODS,
               "coverage": {"metar_airports": int(M.n_airports), "faa_airports": int(P.n_airports),
                            "faa_advisories": int(len(prog)),
                            "faa_advisories_by_kind_status": {f"{k}/{s}": int(n) for (k, s), n in prog.groupby(["kind", "status"]).size().items()},
                            "faa_missing_period": {k: c.n_missing_period for k, c in P.chains.items()},
                            "eval_flights": int(len(ev)), "eval_flown": int(flown["eval"].sum())},
               "ece_tolerance": ECE_TOLERANCE}

    # Committed booking model on the same rows (context)
    m_b, iso_b = bt.fit_calibrated(booking_X["fit"][flown["fit"]], y["fit"], booking_X["val"][flown["val"]], y["val"])
    p_book = iso_b.predict(m_b.predict_proba(booking_X["eval"])[:, 1])
    results["booking"] = runup.score(ev, p_book)
    log("booking", results["booking"]["auc_delayed"], results["booking"]["brier_delayed"])

    for h, sources in TESTS.items():
        arms = ["base"] + [a for a in ("A", "B", "AB") if set(a) <= set(sources)]
        # features per split, cached (they depend only on code and data)
        X = {}
        for k in ("fit", "val", "eval"):
            f = CACHE / f"X_{h}_{k}.parquet"
            if f.exists():
                X[k] = pd.read_parquet(f)
                continue
            base = runup.horizon_features(A, C, R, W, rows[k], h, booking_X[k], W.ok)
            extra = {}
            for name, fn in (("A", M.features), ("B", P.features)):
                if name in sources:
                    extra.update({c: v.astype("float32") for c, v in fn(A, rows[k], HORIZONS[h]).items()})
            X[k] = pd.concat([base, pd.DataFrame(extra, index=base.index)], axis=1)
            X[k].to_parquet(f)
        a_cols = [c for c in X["fit"].columns if c.startswith("mt_")]
        b_cols = [c for c in X["fit"].columns if c.startswith("faa_")]
        base_cols = [c for c in X["fit"].columns if c not in a_cols + b_cols]
        cols = {"base": base_cols, "A": base_cols + a_cols, "B": base_cols + b_cols, "AB": base_cols + a_cols + b_cols}

        r = {"n_features": {a: len(cols[a]) for a in arms}, "new_feature_coverage": {}}
        for grp, gc in (("A", a_cols), ("B", b_cols)):
            if gc:
                r["new_feature_coverage"][grp] = {
                    "origin_covered": round(float(X["eval"][gc[0]].notna().mean()), 4),
                    "both_ends_covered": round(float((X["eval"][f"{'mt' if grp == 'A' else 'faa'}_o_{gc[0].split('_', 2)[2]}"].notna()
                                                      & X["eval"][f"{'mt' if grp == 'A' else 'faa'}_d_{gc[0].split('_', 2)[2]}"].notna()).mean()), 4)}
        if b_cols:
            r["faa_flagged_share_eval"] = {c: round(float(X["eval"][c].mean()), 5) for c in b_cols}
        preds = {}
        for a in arms:
            pf = CACHE / f"pred_{h}_{a}.npy"
            if pf.exists():
                preds[a] = np.load(pf)
                continue
            log(h, "fitting", a, len(cols[a]), "features")
            m, iso = bt.fit_calibrated(X["fit"].loc[flown["fit"], cols[a]], y["fit"], X["val"].loc[flown["val"], cols[a]], y["val"])
            preds[a] = iso.predict(m.predict_proba(X["eval"][cols[a]])[:, 1])
            np.save(pf, preds[a])
        r["arms"] = {a: runup.score(ev, preds[a]) for a in arms}
        for a in arms:
            r["arms"][a].pop("deciles", None)
        r["vs_base"], r["vs_booking"] = {}, {}
        for a in arms[1:]:
            d = runup.brier_diff(ev, preds[a], preds["base"])
            d["months"] = monthly_brier_diff(ev, preds[a], preds["base"])
            d["months_improved"] = int(sum(v < 0 for v in d["months"].values()))
            d["n_months"] = len(d["months"])
            d["auc_change"] = round(r["arms"][a]["auc_delayed"] - r["arms"]["base"]["auc_delayed"], 4)
            d["ece_with"], d["ece_base"] = r["arms"][a]["ece"], r["arms"]["base"]["ece"]
            d["decision"] = decide(d, d["ece_with"], d["ece_base"])
            r["vs_base"][a] = d
        for a in arms:
            r["vs_booking"][a] = runup.brier_diff(ev, preds[a], p_book)
        r["verdict"] = {s: r["vs_base"][s]["decision"]["adopt"] for s in sources}
        results[h] = r
        log(h, {a: (r["arms"][a]["auc_delayed"], r["arms"][a]["brier_delayed"], r["arms"][a]["ece"]) for a in arms},
            {a: (r["vs_base"][a]["diff"], r["vs_base"][a]["ci95"], r["vs_base"][a]["decision"]["adopt"]) for a in arms[1:]})

    results["verdict"] = {s: {h: results[h]["verdict"][s] for h in TESTS if s in results[h]["verdict"]} for s in ("A", "B")}
    results["recommend"] = {s: any(v.values()) for s, v in results["verdict"].items()}
    out = RESULTS / ("b5_metar_faa_quick.json" if QUICK else "b5_metar_faa.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    log("wrote", out, results["verdict"])


if __name__ == "__main__":
    main()
