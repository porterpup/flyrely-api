"""
B5.1: do observed METAR (A) and FAA GS/GDP programs (B) improve run-up delay prediction?  Implements
backtest/PROTOCOL_B5.md with the fixes of backtest/PROTOCOL_B5_1.md (both committed before this script was run;
B5's own run is the earlier revision of this file, PR #12).

    python backtest/experiments_b5.py            # writes backtest/results/b5_1_metar_faa.json
    python backtest/experiments_b5.py --quick    # tiny debug run (2025 months only), writes b5_1_metar_faa_quick.json

H1 holdout confirmation (PROTOCOL_B5_1_HOLDOUT.md; the code below `H1 holdout stages`; without --holdout nothing
below runs and this script behaves exactly as B5.1):

    python backtest/experiments_b5.py --holdout 2026-09 --stage freeze   # no unlock; pre-holdout data only
    FLYRELY_HOLDOUT_UNLOCK=2026-09 python backtest/experiments_b5.py --holdout 2026-09 --stage check
    FLYRELY_HOLDOUT_UNLOCK=2026-09 python backtest/experiments_b5.py --holdout 2026-09 --stage score   # once

Evaluation months 2026-01..08 are validation (PROTOCOL_ADDENDUM_2026-11.md); months >= 2026-09 are never loaded
(runup.load stops at 2026-08 and this script hard-fails on anything later).

Arms per horizon: base (existing run-up horizon model incl. archived-forecast weather), +A, +B, +A+B.
Decision rule per source and horizon: 95% day-block bootstrap CI of Brier(with) - Brier(base) entirely below 0 AND
ECE(with) <= ECE(base) + 0.005.
"""

import argparse
import hashlib
import json
import subprocess
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
import holdout  # noqa: E402  the one holdout policy
import runup  # noqa: E402
from fetch_faa_adv import OUT as FAA_DIR, PROGRAMS_FILE  # noqa: E402
from fetch_metar import busiest_airports  # noqa: E402
from flyrely_features import faa, metar  # noqa: E402
from flyrely_features._common import HORIZONS  # noqa: E402

QUICK = runup.QUICK
CACHE = HERE.parent / "data" / "cache" / ("b5_1_quick" if QUICK else "b5_1")
RESULTS = HERE / "results"
ECE_TOLERANCE = 0.005
FORBIDDEN_FROM = holdout.LOCKED_FROM_TS
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
    f = FAA_DIR / PROGRAMS_FILE
    if not f.exists():
        raise SystemExit(f"{f} missing: run backtest/fetch_faa_adv.py lists, build")
    p = pd.read_parquet(f)
    if p["sent_utc"].max() >= FORBIDDEN_FROM:
        raise RuntimeError("FAA programs contain advisories >= 2026-09 (protected holdout)")
    return p


def check_metar_holdout(M) -> None:
    """Hard-fail when any METAR observation is from the protected holdout (2026-09 on)."""
    if M.ok and M.max_valid is not None and M.max_valid >= FORBIDDEN_FROM:
        raise RuntimeError(f"METAR contains observations >= 2026-09 (protected holdout): {M.max_valid}")


def check_weather_holdout(W) -> None:
    """Hard-fail when any archived-forecast hour is from the protected holdout (2026-09 on)."""
    if W.ok and W.max_utc >= FORBIDDEN_FROM:
        raise RuntimeError(f"weather contains hours >= 2026-09 (protected holdout): {W.max_utc}")


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    df, A = runup.load()
    if df["FlightDate"].max() >= FORBIDDEN_FROM:
        raise RuntimeError("protected holdout data (>= 2026-09) present")
    W = runup.Weather(A["airports"])
    check_weather_holdout(W)
    C, R = runup.build_counters(A), runup.Rotation(A)
    M = metar.Metar(A["airports"])
    check_metar_holdout(M)
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

    results = {"protocol": "backtest/PROTOCOL_B5_1.md (on PROTOCOL_B5.md)", "quick_debug_run": QUICK, "periods": PERIODS,
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
    out = RESULTS / ("b5_1_metar_faa_quick.json" if QUICK else "b5_1_metar_faa.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    log("wrote", out, results["verdict"])


# =============================================================================
# H1 holdout stages (PROTOCOL_B5_1_HOLDOUT.md sections 5 and 8). Nothing below runs without --holdout.
# =============================================================================

MODELS_DIR = holdout.MODELS_DIR
MANIFEST = RESULTS / "h1_models_manifest.json"
B5_1_RESULTS = RESULTS / "b5_1_metar_faa.json"
PROTOCOL_H1 = "backtest/PROTOCOL_B5_1_HOLDOUT.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=HERE.parent, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "flyrely_features", "backtest"))}


def classify_outcome(diff: dict, ece_with: float, ece_base: float, tol: float = ECE_TOLERANCE) -> str:
    """The three outcomes of PROTOCOL_B5_1_HOLDOUT.md section 4.

    confirmed: the 95% CI of the Brier difference is entirely below 0 AND ECE(with) <= ECE(base) + tol.
    contradicted: otherwise, when the point estimate is at or above 0 OR the ECE condition fails.
    not_confirmed: otherwise (rule fails, point estimate below 0, ECE fine)."""
    ece_ok = ece_with <= ece_base + tol
    if diff["ci95"][1] < 0 and ece_ok:
        return "confirmed"
    if diff["diff"] >= 0 or not ece_ok:
        return "contradicted"
    return "not_confirmed"


def weekly_brier_diff(frame: pd.DataFrame, p_a: np.ndarray, p_b: np.ndarray) -> dict:
    """Brier(a) - Brier(b) per 7-day block of the month (4 full weeks plus the remainder), flown flights only."""
    flown = ~frame["cancelled"].to_numpy()
    y = frame["delayed"].to_numpy()
    first = frame["FlightDate"].min().normalize().replace(day=1)
    week = ((frame["FlightDate"] - first).dt.days // 7).to_numpy()
    out = {}
    for k in sorted(set(week)):
        i = np.flatnonzero((week == k) & flown)
        days = frame["FlightDate"].iloc[i]
        out[f"wk{k + 1} {days.min():%m-%d}..{days.max():%m-%d}"] = round(
            float(np.mean((p_a[i] - y[i]) ** 2) - np.mean((p_b[i] - y[i]) ** 2)), 6)
    return out


def load_manifest(path: Path = None) -> dict:
    path = path or MANIFEST
    if not path.exists():
        raise SystemExit(f"{path} missing: run `--stage freeze` first (the models must be frozen before the holdout opens)")
    return json.loads(path.read_text())


def verify_models(manifest: dict, models_dir: Path = None) -> None:
    """Every model file must exist and match the SHA-256 recorded at freeze time."""
    models_dir = models_dir or MODELS_DIR
    bad = []
    for name, m in manifest["models"].items():
        f = models_dir / m["file"]
        if not f.exists():
            bad.append(f"{name}: {f} missing")
        elif sha256_file(f) != m["sha256"]:
            bad.append(f"{name}: hash mismatch ({f})")
    if bad:
        raise SystemExit("frozen models do not match the manifest:\n  " + "\n  ".join(bad))


TREE_PATHS = ["flyrely_features", "backtest/runup.py", "backtest/backtest.py", "backtest/experiments_b5.py",
              "backtest/holdout.py", "backtest/fetch_faa_adv.py"]       # code that must equal the freeze commit's


def _git(*args):
    return subprocess.run(["git", *args], cwd=HERE.parent, capture_output=True, text=True)


def verify_frozen_code(manifest: dict) -> None:
    """Refuse a dirty tree, and refuse unless the feature/run code equals the code of the manifest's freeze commit
    (compared by git tree/blob hash, so later commits that touch other files, such as the report, are fine)."""
    if _git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise SystemExit("score refused: the working tree has uncommitted changes to tracked files")
    commit = manifest["code"]["commit"]
    bad = []
    for p in TREE_PATHS:
        a, b = _git("rev-parse", f"HEAD:{p}"), _git("rev-parse", f"{commit}:{p}")
        if a.returncode or b.returncode or a.stdout.strip() != b.stdout.strip():
            bad.append(p)
    if bad:
        raise SystemExit(f"score refused: {bad} differ from the freeze commit {commit[:10]}. "
                         "Re-run `--stage freeze` at the merged code and commit the new manifest first")


def verify_airports(manifest: dict) -> None:
    if not holdout.AIRPORTS_FILE.exists() or sha256_file(holdout.AIRPORTS_FILE) != manifest.get("airports_file_sha256"):
        raise SystemExit("score refused: h1_airports.json does not match the manifest (airport lists are frozen)")


def hash_holdout_files(month: str) -> dict:
    d = holdout.holdout_dir(month)
    return {str(p.relative_to(d)): sha256_file(p) for p in sorted(d.rglob("*")) if p.is_file()} if d.exists() else {}


def check_marker_path(month: str, results_dir: Path = None) -> Path:
    return (results_dir or RESULTS) / f"h1_check_{month}.pass.json"


def verify_check_marker(month: str, results_dir: Path = None) -> None:
    """Stage 2a must have passed on exactly the data files that are on disk now."""
    f = check_marker_path(month, results_dir)
    if not f.exists():
        raise SystemExit(f"score refused: no passing stage-2a marker ({f.name}); run `--stage check` until it passes")
    if json.loads(f.read_text())["files"] != hash_holdout_files(month):
        raise SystemExit("score refused: the holdout data changed since `check` passed; repeat `--stage check`")


def preflight_score(month: str, force: bool = False, reason: str = None, results_dir: Path = None,
                    manifest_path: Path = None, models_dir: Path = None) -> tuple[dict, Path]:
    """Everything `score` must satisfy before it touches any holdout data. Returns (manifest, result path).

    The original result file is never overwritten: a re-run (only with --force-rerun and a --reason) writes
    h1_b5_1_<month>_rerun<N>.json next to it."""
    holdout.require_unlock(month)
    manifest = load_manifest(manifest_path)
    if manifest.get("holdout_month") != month:
        raise SystemExit(f"score refused: the manifest is for {manifest.get('holdout_month')}, not {month}")
    verify_models(manifest, models_dir)
    verify_airports(manifest)
    verify_frozen_code(manifest)
    verify_check_marker(month, results_dir)
    results_dir = results_dir or RESULTS
    orig = results_dir / f"h1_b5_1_{month}.json"
    attempts = results_dir / f"h1_b5_1_{month}.attempts.json"
    n = len(json.loads(attempts.read_text())) if attempts.exists() else 0
    if not (n or orig.exists()):
        if force:
            raise SystemExit("--force-rerun refused: there is no earlier attempt to re-run")
        return manifest, orig
    if not force:
        raise SystemExit(f"{attempts.name if n else orig.name} exists: the holdout is scored once (an attempt is recorded "
                         "before any data is loaded). --force-rerun --reason exists only for the bug-fix case "
                         "of the protocol (section 4, item 3)")
    if not (reason or "").strip():
        raise SystemExit("--force-rerun needs --reason \"<why>\" (it is recorded in the result)")
    k = max(n, 1)
    while (results_dir / f"h1_b5_1_{month}_rerun{k}.json").exists():
        k += 1
    log(f"FORCE-RERUN {k}: {reason} (protocol section 4, item 3: the original result is kept, both are reported)")
    return manifest, results_dir / f"h1_b5_1_{month}_rerun{k}.json"


def record_attempt(month: str, out: Path, manifest: dict, reason: str = None, results_dir: Path = None) -> None:
    """Written BEFORE any holdout data is loaded, so an interrupted run still counts as the one attempt."""
    f = (results_dir or RESULTS) / f"h1_b5_1_{month}.attempts.json"
    rec = json.loads(f.read_text()) if f.exists() else []
    rec.append({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "output": out.name,
                "code_commit": _git("rev-parse", "HEAD").stdout.strip(), "reason": reason,
                "models_sha256": {k: v["sha256"] for k, v in manifest["models"].items()}})
    tmp = f.with_suffix(".part")
    tmp.write_text(json.dumps(rec, indent=2) + "\n")
    tmp.rename(f)


def horizon_X(A, C, R, W, M, P, rows, h, booking_X, sources) -> pd.DataFrame:
    """Features of one horizon, built exactly as `main` builds them."""
    base = runup.horizon_features(A, C, R, W, rows, h, booking_X, W.ok)
    extra = {}
    for name, fn in (("A", M.features), ("B", P.features)):
        if name in sources:
            extra.update({c: v.astype("float32") for c, v in fn(A, rows, HORIZONS[h]).items()})
    return pd.concat([base, pd.DataFrame(extra, index=base.index)], axis=1)


def arm_columns(columns) -> tuple[dict, list, list]:
    a_cols = [c for c in columns if c.startswith("mt_")]
    b_cols = [c for c in columns if c.startswith("faa_")]
    base_cols = [c for c in columns if c not in a_cols + b_cols]
    return ({"base": base_cols, "A": base_cols + a_cols, "B": base_cols + b_cols, "AB": base_cols + a_cols + b_cols},
            a_cols, b_cols)


def predict_with(model_file: Path, X: pd.DataFrame) -> np.ndarray:
    import joblib
    obj = joblib.load(model_file)
    return obj["iso"].predict(obj["model"].predict_proba(X[obj["columns"]])[:, 1])


def evaluate_horizon(h, sources, cols, a_cols, b_cols, X_eval, ev, preds, p_book, holdout_mode: bool) -> dict:
    """The per-horizon block of the result file, as in `main`. holdout_mode swaps "months improved" for per-week
    differences and adds the three-outcome label."""
    arms = ["base"] + [a for a in ("A", "B", "AB") if set(a) <= set(sources)]
    r = {"n_features": {a: len(cols[a]) for a in arms}, "new_feature_coverage": {}}
    for grp, gc in (("A", a_cols), ("B", b_cols)):
        if gc:
            r["new_feature_coverage"][grp] = {
                "origin_covered": round(float(X_eval[gc[0]].notna().mean()), 4),
                "both_ends_covered": round(float((X_eval[f"{'mt' if grp == 'A' else 'faa'}_o_{gc[0].split('_', 2)[2]}"].notna()
                                                  & X_eval[f"{'mt' if grp == 'A' else 'faa'}_d_{gc[0].split('_', 2)[2]}"].notna()).mean()), 4)}
    if b_cols:
        r["faa_flagged_share_eval"] = {c: round(float(X_eval[c].mean()), 5) for c in b_cols}
    r["arms"] = {a: runup.score(ev, preds[a]) for a in arms}
    for a in arms:
        r["arms"][a].pop("deciles", None)
    r["vs_base"], r["vs_booking"] = {}, {}
    for a in arms[1:]:
        d = runup.brier_diff(ev, preds[a], preds["base"])
        if holdout_mode:
            d["weeks"] = weekly_brier_diff(ev, preds[a], preds["base"])
        else:
            d["months"] = monthly_brier_diff(ev, preds[a], preds["base"])
            d["months_improved"] = int(sum(v < 0 for v in d["months"].values()))
            d["n_months"] = len(d["months"])
        d["auc_change"] = round(r["arms"][a]["auc_delayed"] - r["arms"]["base"]["auc_delayed"], 4)
        d["ece_with"], d["ece_base"] = r["arms"][a]["ece"], r["arms"]["base"]["ece"]
        d["decision"] = decide(d, d["ece_with"], d["ece_base"])
        if holdout_mode:
            d["outcome"] = classify_outcome(d, d["ece_with"], d["ece_base"])
        r["vs_base"][a] = d
    for a in arms:
        r["vs_booking"][a] = runup.brier_diff(ev, preds[a], p_book)
    r["verdict"] = {s: r["vs_base"][s]["decision"]["adopt"] for s in sources}
    if holdout_mode:
        r["outcome"] = {s: r["vs_base"][s]["outcome"] for s in sources}
    return r


def build_context(month: str = None) -> dict:
    """Data and feature builders. month=None: the B5.1 pre-holdout data, with every B5.1 guard. month=YYYY-MM:
    that unlocked month added from data/holdout/YYYY-MM/ (BTS, METAR, weather, FAA), with window guards."""
    hd = holdout.holdout_dir(month) if month else None
    df, A = runup.load(end=month, holdout_dir=(hd / "bts") if month else None)
    W = runup.Weather(A["airports"], weather_dirs=[hd / "weather"] if month else ())
    C, R = runup.build_counters(A), runup.Rotation(A)
    M = metar.Metar(A["airports"], metar_dirs=[hd / "metar"] if month else ())
    prog = load_programs()
    if not month:
        if df["FlightDate"].max() >= FORBIDDEN_FROM:
            raise RuntimeError("protected holdout data (>= 2026-09) present")
        check_weather_holdout(W)
        check_metar_holdout(M)
        covered = busiest_airports(100)
    else:
        _, last = holdout.input_window(month)
        end = pd.Timestamp(last) + pd.Timedelta(days=1)          # inputs stop at the end of the window's last day
        f = hd / "faa_adv" / "programs.parquet"
        if not f.exists():
            raise SystemExit(f"{f} missing: run fetch_faa_adv.py lists, build --holdout {month}")
        prog = pd.concat([prog, pd.read_parquet(f)], ignore_index=True)
        if df["FlightDate"].max() > pd.Timestamp(holdout.month_bounds(month)[1]):
            raise RuntimeError("BTS rows after the holdout month (refused)")
        if W.ok and W.max_utc >= end or (M.ok and M.max_valid is not None and M.max_valid >= end) or prog["sent_utc"].max() >= end:
            raise RuntimeError(f"inputs run past the {month} window ({last}); refused")
        covered = holdout.frozen_airports("faa_covered")
    P = faa.Programs(prog, A["airports"], covered)
    return {"df": df, "A": A, "W": W, "C": C, "R": R, "M": M, "P": P, "prog": prog, "covered": covered}


def deep_diff(ref, new, path="") -> list:
    """Paths where two JSON-like structures differ (exact equality)."""
    if isinstance(ref, dict) and isinstance(new, dict):
        out = [f"{path}/{k}: missing" for k in ref.keys() - new.keys()] + [f"{path}/{k}: extra" for k in new.keys() - ref.keys()]
        for k in ref.keys() & new.keys():
            out += deep_diff(ref[k], new[k], f"{path}/{k}")
        return out
    if isinstance(ref, list) and isinstance(new, list):
        if len(ref) != len(new):
            return [f"{path}: length {len(ref)} != {len(new)}"]
        return [d for i, (a, b) in enumerate(zip(ref, new)) for d in deep_diff(a, b, f"{path}[{i}]")]
    return [] if ref == new else [f"{path}: {ref!r} != {new!r}"]


def stage_freeze(month: str, models_dir: Path = None, manifest_path: Path = None) -> None:
    """Rebuild the B5.1 features, fit every arm, re-predict validation and prove it equals B5.1; then save the
    models and the manifest. No unlock. Refused when any holdout data is present."""
    import joblib
    if runup.QUICK:
        raise SystemExit("--quick has no meaning for the holdout stages")
    if not holdout.is_month(month) or not holdout.is_locked_month(month):
        raise SystemExit(f"--holdout {month}: expected a protected month (>= {holdout.LOCKED_FROM})")
    present = holdout.holdout_present(month) + holdout.locked_files_in_main()
    if present:
        raise SystemExit(f"freeze refused: holdout data is already present: {present[:5]}")
    models_dir, manifest_path = models_dir or MODELS_DIR, manifest_path or MANIFEST
    # always from scratch: a fresh run folder (never reuse an earlier cache; nothing under data/ is deleted)
    cache = HERE.parent / "data" / "cache" / f"b5_1_holdout_{month}" / time.strftime("run_%Y%m%d_%H%M%S")
    cache.mkdir(parents=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    ref = json.loads(B5_1_RESULTS.read_text())

    ctx = build_context(None)
    df, A, W, C, R, M, P, prog, covered = (ctx[k] for k in ("df", "A", "W", "C", "R", "M", "P", "prog", "covered"))
    airports = {"metar": sorted(f.stem.rsplit("_", 1)[0] for f in (HERE.parent / "data" / "metar").glob("*_2026.csv")),
                "weather": sorted(f.stem for f in (HERE.parent / "data" / "weather").glob("*.parquet")),
                "faa_covered": list(covered),
                "note": "frozen at stage 1 of PROTOCOL_B5_1_HOLDOUT.md; the holdout fetchers read this, never recompute"}
    log(f"METAR airports {M.n_airports if M.ok else 0}; FAA airports {P.n_airports}; advisories {len(prog)}; "
        f"weather files {len(airports['weather'])}")

    PERIODS = {"fit": runup.PERIODS["fit"], "val": runup.PERIODS["val"], "eval": runup.PERIODS["holdout"]}
    within = lambda a, b: np.flatnonzero(((df["ym"] >= a) & (df["ym"] <= b)).to_numpy())
    rows = {k: within(*v["rows"]) for k, v in PERIODS.items()}
    frame = {k: df.iloc[r] for k, r in rows.items()}
    booking_X = {k: bt.booking_features(frame[k], df.iloc[within(*v["rates"])]).astype("float32") for k, v in PERIODS.items()}
    flown = {k: ~frame[k]["cancelled"].to_numpy() for k in rows}
    y = {k: frame[k]["delayed"].to_numpy()[flown[k]] for k in rows}
    ev = frame["eval"]

    results = {"protocol": "backtest/PROTOCOL_B5_1.md (on PROTOCOL_B5.md)", "quick_debug_run": False, "periods": PERIODS,
               "coverage": {"metar_airports": int(M.n_airports), "faa_airports": int(P.n_airports),
                            "faa_advisories": int(len(prog)),
                            "faa_advisories_by_kind_status": {f"{k}/{s}": int(n) for (k, s), n in prog.groupby(["kind", "status"]).size().items()},
                            "faa_missing_period": {k: c.n_missing_period for k, c in P.chains.items()},
                            "eval_flights": int(len(ev)), "eval_flown": int(flown["eval"].sum())},
               "ece_tolerance": ECE_TOLERANCE}
    manifest = {"protocol": PROTOCOL_H1, "holdout_month": month, "models": {}, "feature_columns": {},
                "b5_1_reproduction": {"reference": "backtest/results/b5_1_metar_faa.json",
                                      "reference_sha256": sha256_file(B5_1_RESULTS), "feature_frames_equal": {},
                                      "cached_predictions_equal": {}}}
    problems = []

    # committed booking model, on the validation rows (context arm)
    m_b, iso_b = bt.fit_calibrated(booking_X["fit"][flown["fit"]], y["fit"], booking_X["val"][flown["val"]], y["val"])
    joblib.dump({"model": m_b, "iso": iso_b, "columns": list(booking_X["fit"].columns)}, models_dir / "booking.joblib")
    manifest["models"]["booking"] = {"file": "booking.joblib", "sha256": sha256_file(models_dir / "booking.joblib"),
                                     "n_features": booking_X["fit"].shape[1]}
    manifest["feature_columns"]["booking"] = list(booking_X["fit"].columns)
    p_book = predict_with(models_dir / "booking.joblib", booking_X["eval"])
    results["booking"] = runup.score(ev, p_book)
    log("booking", results["booking"]["auc_delayed"], results["booking"]["brier_delayed"])

    for h, sources in TESTS.items():
        X = {}
        for k in ("fit", "val", "eval"):
            X[k] = horizon_X(A, C, R, W, M, P, rows[k], h, booking_X[k], sources)
            X[k].to_parquet(cache / f"X_{h}_{k}.parquet")
            old = CACHE / f"X_{h}_{k}.parquet"
            try:
                pd.testing.assert_frame_equal(X[k], pd.read_parquet(old), check_exact=True)
                manifest["b5_1_reproduction"]["feature_frames_equal"][f"{h}_{k}"] = True
            except AssertionError as e:
                manifest["b5_1_reproduction"]["feature_frames_equal"][f"{h}_{k}"] = False
                problems.append(f"features X_{h}_{k} differ from the B5.1 cache: {str(e)[:300]}")
        cols, a_cols, b_cols = arm_columns(X["fit"].columns)
        manifest["feature_columns"][h] = list(X["fit"].columns)
        arms = ["base"] + [a for a in ("A", "B", "AB") if set(a) <= set(sources)]
        preds = {}
        for a in arms:
            log(h, "fitting", a, len(cols[a]), "features")
            m, iso = bt.fit_calibrated(X["fit"].loc[flown["fit"], cols[a]], y["fit"], X["val"].loc[flown["val"], cols[a]], y["val"])
            f = models_dir / f"{h}_{a}.joblib"
            joblib.dump({"model": m, "iso": iso, "columns": cols[a]}, f)
            manifest["models"][f"{h}_{a}"] = {"file": f.name, "sha256": sha256_file(f), "n_features": len(cols[a])}
            preds[a] = predict_with(f, X["eval"])                    # from the saved file, as `score` will do
            pf = CACHE / f"pred_{h}_{a}.npy"
            same = bool(pf.exists() and np.array_equal(preds[a], np.load(pf)))
            manifest["b5_1_reproduction"]["cached_predictions_equal"][f"{h}_{a}"] = same
            if not same:
                problems.append(f"predictions {h}_{a} differ from the B5.1 cache")
        results[h] = evaluate_horizon(h, sources, cols, a_cols, b_cols, X["eval"], ev, preds, p_book, holdout_mode=False)
        del X
    results["verdict"] = {s: {h: results[h]["verdict"][s] for h in TESTS if s in results[h]["verdict"]} for s in ("A", "B")}
    results["recommend"] = {s: any(v.values()) for s, v in results["verdict"].items()}

    new = json.loads(json.dumps(results, default=float))
    diffs = deep_diff(ref, {k: new[k] for k in ref if k in new}) + [f"{k}: not produced" for k in ref if k not in new]
    manifest["b5_1_reproduction"]["results_equal"] = not diffs
    manifest["b5_1_reproduction"]["n_differences"] = len(diffs)
    if diffs or problems:
        raise SystemExit("STOP: the frozen models are NOT the B5.1 models. Nothing was adjusted and no manifest was written.\n"
                         + "\n".join((problems + diffs)[:40]))
    manifest.update({"code": git_state(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "note": "code.commit is the commit checked out at freeze time; the protocol's code-freeze hash is the "
                             "merge commit of the holdout tooling"})
    old = holdout.AIRPORTS_FILE
    if old.exists() and json.loads(old.read_text()) != airports:
        raise SystemExit(f"{old} exists and differs from the airports found now: the lists are frozen, not overwritten")
    old.write_text(json.dumps(airports, indent=2) + "\n")
    manifest["airports_file_sha256"] = sha256_file(old)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log("B5.1 reproduced exactly; wrote", manifest_path, "and", old)


def stage_check(month: str, models_dir: Path = None, manifest_path: Path = None) -> bool:
    """Stage 2a integrity checks. Prints PASS/FAIL lines only; nothing here uses a delay outcome."""
    import pyarrow.parquet as pq
    from fetch_faa_adv import holdout_out, parse_list
    from fetch_weather import LEADS, VARIABLES
    holdout.require_unlock(month)
    hd = holdout.holdout_dir(month)
    first, last = holdout.input_window(month)
    mfirst, mlast = holdout.month_bounds(month)
    results = []

    def report(name, ok, detail=""):
        results.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)

    # BTS
    f = hd / "bts" / f"{month}.parquet"
    if not f.exists():
        report("bts file", False, f"{f} missing")
    else:
        d = pd.read_parquet(f, columns=["FlightDate"])["FlightDate"]
        n_days = d.dt.normalize().nunique()
        per_day = d.dt.normalize().value_counts()
        prev_f = bt.PARQUET / f"{int(month[:4]) - 1}-{month[5:]}.parquet"
        prev_n = len(pd.read_parquet(prev_f, columns=["Origin"])) if prev_f.exists() else None
        need = set(pq.read_schema(prev_f).names) if prev_f.exists() else set()
        have = set(pq.read_schema(f).names)
        n_expected = (mlast - mfirst).days + 1
        report("bts distinct flight dates", n_days == n_expected, f"{n_days} of {n_expected}")
        report("bts rows per day >= 10,000", bool((per_day >= 10_000).all()) and n_days == n_expected, f"min {int(per_day.min())}")
        report("bts total rows within +-15% of last year", prev_n is not None and abs(len(d) / prev_n - 1) <= 0.15,
               f"{len(d)} vs {prev_n}")
        report("bts required columns", bool(need) and need <= have, f"missing {sorted(need - have)}")
        report("bts no date outside the month", d.min() >= pd.Timestamp(mfirst) and d.max() <= pd.Timestamp(mlast),
               f"{d.min().date()}..{d.max().date()}")

    # METAR files (all frozen airports, every day of the month)
    days = {str(x.date()) for x in pd.date_range(mfirst, mlast)}
    short = {}
    for ap_ in holdout.frozen_airports("metar"):
        mf = hd / "metar" / f"{ap_}_{month}.csv"
        if not mf.exists():
            short[ap_] = "no file"
            continue
        got = set(pd.read_csv(mf, usecols=["valid"], dtype=str)["valid"].dropna().str[:10]) & days
        if got != days:
            short[ap_] = f"{len(got)}/{len(days)} days"
    report("metar: every airport has a report on every day", not short, f"{len(short)} short: {dict(list(short.items())[:5])}")

    # FAA list pages and programs
    lst_dir = holdout_out(month) / "list"
    want = [str(x.date()) for x in pd.date_range(first, last)]
    missing, unparseable = [], []
    for dstr in want:
        p = lst_dir / f"{dstr}.html"
        if not p.exists():
            missing.append(dstr)
            continue
        txt = p.read_text()
        try:
            parse_list(txt)
            if len(txt) <= 500:
                raise ValueError("page too short")
        except Exception:
            unparseable.append(dstr)
    report("faa list pages present and parseable", not missing and not unparseable,
           f"{len(want) - len(missing)} of {len(want)} present, unparseable {unparseable[:3]}, missing {missing[:3]}")
    pf = holdout_out(month) / "programs.parquet"
    if pf.exists():
        pr = pd.read_parquet(pf)
        in_month = pr[(pr["sent_utc"] >= pd.Timestamp(mfirst)) & (pr["sent_utc"] < pd.Timestamp(mlast) + pd.Timedelta(days=1))]
        n = in_month["sent_utc"].dt.normalize().nunique()
        report("faa GS/GDP advisories on >= 25 days", n >= 25, f"{n} days")
    else:
        report("faa programs file", False, f"{pf} missing")

    # weather: every airport, every hour
    hours = pd.date_range(pd.Timestamp(first), pd.Timestamp(last) + pd.Timedelta(hours=23), freq="h")
    cols = [f"{v}_previous_day{d}" for v in VARIABLES for d in LEADS]
    bad = {}
    for ap_ in holdout.frozen_airports("weather"):
        wf = hd / "weather" / f"{ap_}.parquet"
        if not wf.exists():
            bad[ap_] = "no file"
            continue
        w = pd.read_parquet(wf)
        if set(w["utc_hour"]) != set(hours) or w["utc_hour"].duplicated().any():
            bad[ap_] = f"hours {w['utc_hour'].nunique()}/{len(hours)}"
        elif any(c not in w or w[c].isna().any() for c in cols):
            bad[ap_] = "missing forecast values"
    report("weather: every airport and hour", not bad, f"{len(bad)} bad: {dict(list(bad.items())[:5])}")

    # frozen models
    try:
        verify_models(load_manifest(manifest_path), models_dir)
        report("frozen models match the manifest", True)
    except SystemExit as e:
        report("frozen models match the manifest", False, str(e))

    # origin coverage of holdout flights (needs the loaders; no outcomes are read)
    ctx = build_context(month)
    df, A, M = ctx["df"], ctx["A"], ctx["M"]
    rows = np.flatnonzero((df["ym"] == month).to_numpy())
    for h in ("3h", "1h"):
        cov = float(pd.Series(M.features(A, rows, HORIZONS[h])["mt_o_ceiling_ft"]).notna().mean())
        report(f"metar origin coverage of holdout flights at {h} >= 85%", cov >= 0.85, f"{cov:.1%}")
    ok = all(results)
    if ok:
        mf = check_marker_path(month)
        mf.write_text(json.dumps({"month": month, "files": hash_holdout_files(month),
                                  "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
    elif check_marker_path(month).exists():
        check_marker_path(month).unlink()          # a failed check invalidates an earlier pass
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED: fix the data and repeat stage 2a (no score has been computed)")
    return ok


def stage_score(month: str, force: bool = False, reason: str = None) -> None:
    """The one scoring run: frozen models (hashes checked), holdout features, decide. Never fits."""
    manifest, out = preflight_score(month, force, reason)
    record_attempt(month, out, manifest, reason)
    ctx = build_context(month)
    df, A, W, C, R, M, P = (ctx[k] for k in ("df", "A", "W", "C", "R", "M", "P"))
    rows = np.flatnonzero((df["ym"] == month).to_numpy())
    if not len(rows):
        raise SystemExit(f"no {month} flights loaded")
    ev = df.iloc[rows]
    flown = ~ev["cancelled"].to_numpy()
    y1 = pd.Period(month, "M") - 12
    windows = {"rates_12m_before": (str(y1), str(pd.Period(month, "M") - 1)),
               "rates_b5_1_window": ("2025-01", "2025-12")}
    within = lambda a, b: np.flatnonzero(((df["ym"] >= a) & (df["ym"] <= b)).to_numpy())
    booking_X = {k: bt.booking_features(ev, df.iloc[within(*w)]).astype("float32") for k, w in windows.items()}
    results = {"protocol": PROTOCOL_H1, "holdout_month": month,
               "force_rerun": {"reason": reason, "output": out.name} if force else None,
               "code": git_state(), "manifest_sha256": sha256_file(MANIFEST), "models": manifest["models"],
               "windows": windows,
               "coverage": {"metar_airports": int(M.n_airports), "faa_airports": int(P.n_airports),
                            "faa_advisories_by_kind_status": {f"{k}/{s}": int(n) for (k, s), n in ctx["prog"].groupby(["kind", "status"]).size().items()},
                            "eval_flights": int(len(ev)), "eval_flown": int(flown.sum()),
                            "flight_dates": int(ev["FlightDate"].nunique())},
               "ece_tolerance": ECE_TOLERANCE}
    bk = MODELS_DIR / manifest["models"]["booking"]["file"]
    p_book = predict_with(bk, booking_X["rates_12m_before"])
    results["booking"] = runup.score(ev, p_book)
    gated = {}
    for h, sources in TESTS.items():
        arms = ["base"] + [a for a in ("A", "B", "AB") if set(a) <= set(sources)]
        X = horizon_X(A, C, R, W, M, P, rows, h, booking_X["rates_12m_before"], sources)
        if list(X.columns) != manifest["feature_columns"][h]:
            raise SystemExit(f"{h}: holdout feature columns differ from the frozen ones")
        cols, a_cols, b_cols = arm_columns(X.columns)
        preds = {a: predict_with(MODELS_DIR / manifest["models"][f"{h}_{a}"]["file"], X) for a in arms}
        results[h] = evaluate_horizon(h, sources, cols, a_cols, b_cols, X, ev, preds, p_book, holdout_mode=True)
        # non-gating sensitivity: the B5.1 trailing-rate window (2025-01..2025-12)
        Xs = horizon_X(A, C, R, W, M, P, rows, h, booking_X["rates_b5_1_window"], sources)
        ps = {a: predict_with(MODELS_DIR / manifest["models"][f"{h}_{a}"]["file"], Xs) for a in arms}
        sens = {"arms": {a: {k: v for k, v in runup.score(ev, ps[a]).items() if k in ("auc_delayed", "brier_delayed", "ece")}
                         for a in arms}, "vs_base": {}}
        for a in arms[1:]:
            d = runup.brier_diff(ev, ps[a], ps["base"])
            d["ece_with"], d["ece_base"] = sens["arms"][a]["ece"], sens["arms"]["base"]["ece"]
            d["outcome"] = classify_outcome(d, d["ece_with"], d["ece_base"])
            sens["vs_base"][a] = d
        results[h]["sensitivity_b5_1_rate_window"] = sens
        for s in sources:
            d = results[h]["vs_base"][s]
            gated[f"H1-{s}-{h}"] = {"brier_diff": d["diff"], "ci95": d["ci95"], "ece_with": d["ece_with"],
                                    "ece_base": d["ece_base"], "outcome": d["outcome"]}
    results["gated"] = gated
    results["confirmed_horizons"] = {s: [h for h in TESTS if s in TESTS[h] and results[h]["outcome"][s] == "confirmed"]
                                     for s in ("A", "B")}
    out.write_text(json.dumps(results, indent=2, default=float))
    log("wrote", out, {k: v["outcome"] for k, v in gated.items()})
    log("now check that the unlock is not set: env | grep FLYRELY_HOLDOUT || echo OK")


def holdout_cli(argv=None) -> None:
    ap = argparse.ArgumentParser(description="H1 holdout stages (PROTOCOL_B5_1_HOLDOUT.md)")
    ap.add_argument("--holdout", metavar="YYYY-MM", required=True)
    ap.add_argument("--stage", choices=["freeze", "check", "score"], required=True)
    ap.add_argument("--force-rerun", action="store_true", help="score: only for the protocol's bug-fix case (section 4); "
                    "writes h1_b5_1_<month>_rerun<N>.json and never replaces the original")
    ap.add_argument("--reason", help="score --force-rerun: why (recorded in the result)")
    a = ap.parse_args(argv)
    if (a.force_rerun or a.reason) and a.stage != "score":
        raise SystemExit("--force-rerun/--reason apply to --stage score only")
    if a.reason and not a.force_rerun:
        raise SystemExit("--reason is for --force-rerun")
    RESULTS.mkdir(exist_ok=True)
    if a.stage == "freeze":
        stage_freeze(a.holdout)
    elif a.stage == "check":
        sys.exit(0 if stage_check(a.holdout) else 1)
    else:
        stage_score(a.holdout, a.force_rerun, a.reason)


if __name__ == "__main__":
    if any(x.startswith(("--holdout", "--stage", "--force-rerun")) for x in sys.argv[1:]):
        holdout_cli()
    else:
        main()
