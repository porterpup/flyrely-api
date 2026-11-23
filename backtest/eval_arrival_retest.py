"""
Recalibrated arrival re-test on September 2026: Amendment 1 of backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md.

    python backtest/eval_arrival_retest.py --stage freeze    # fit the level offsets a_t on 2026-03..08, pin hashes
    python backtest/eval_arrival_retest.py --stage check     # integrity checks on the September files (no outcomes)
    python backtest/eval_arrival_retest.py --stage score     # the one scoring run
    python backtest/eval_arrival_retest.py --stage score --force-rerun --reason "<bug fixed>"

Arm L-S (D1): for each arrival threshold t, logit(p') = logit(p_L) + a_t with the slope fixed at 1, then the existing
rearrangement. a_t is the unweighted maximum-likelihood offset on the clarification-12 labels of the trailing six
months before the evaluation month (diverted = late), p clipped to [1e-6, 1-1e-6]. Nothing else is fitted: the
ladder, the 2025 isotonic shape, the production model and the connection model stay frozen.

Only the freeze stage fits anything. It also stores what the gates and the C confirmation need that comes from
before September (the 2025-01..06 climatology ladder and the 2025-07..12 B-L table), so `score` fits nothing.
The functions of eval_arrival_connections.py are reused unchanged (that file is not edited). The A1-A4 decision block
is copied statement for statement from `arrival_evaluation` there, because it is inline in that function; thresholds
are unchanged.

Nothing here prints a September outcome number except the final scoring run's result summary (outcomes only after
the attempt record is written). FLYRELY_HOLDOUT_UNLOCK must not be set.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import arrival_data as ad  # noqa: E402
import eval_arrival_connections as E  # noqa: E402
import pairs as pr  # noqa: E402
from flyrely_dist import ladder as L  # noqa: E402
from flyrely_dist.connections import DependenceModel  # noqa: E402

PROTOCOL = "backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md (Amendment 1, 2026-11-22)"
RESULTS = HERE / "results"
EVAL_MONTH = "2026-09"
FIT_WINDOW_MONTHS = 6
FREEZE_PATH = RESULTS / "arrival_retest_freeze.json"
ATTEMPTS_NAME = f"arrival_retest_{EVAL_MONTH}.attempts.json"
OUT_NAME = f"arrival_retest_{EVAL_MONTH}.json"
MARKER_NAME = f"arrival_retest_check_{EVAL_MONTH}.pass.json"
EXPECTED_ROWS = 630_177                         # parquet_v2/2026-09 (published in the H1 report)
CLIMATOLOGY_MONTHS = E.FIT_MONTHS               # 2025-01..06, as in the original evaluation
BL_MONTHS = ("2025-07", "2025-12")              # B-L table months, as in the original evaluation
CONFIG = {"frac": 1.0}                          # E.score_month reads cfg["frac"] only

MODEL_FILES = {                                 # name -> file under models/
    "arrival_ladder": "arrival_ladder_v1.joblib",
    "production_model": "flight_delay_model.joblib",
    "connection_dependence": "connection_dependence_v1.json",
    "mct_defaults": "mct_defaults.csv",
}
# Code that must equal the freeze commit's (compared by git blob/tree hash at score time).
CODE_PATHS = ["flyrely_features", "flyrely_dist", "main.py", "arrival_service.py",
              "backtest/eval_arrival_retest.py", "backtest/eval_arrival_connections.py", "backtest/arrival_data.py",
              "backtest/pairs.py", "backtest/prepare_arrival.py", "backtest/prepare.py", "backtest/backtest.py",
              "backtest/runup.py", "backtest/experiments_b5.py", "backtest/holdout.py"]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ small helpers

def sha256_file(path) -> str:
    return E.sha256_file(path)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def refuse_if_unlock_set(env=None) -> None:
    if ad.UNLOCK_VAR in (os.environ if env is None else env):
        sys.exit(f"{ad.UNLOCK_VAR} is set; the arrival re-test never runs with it")


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def fit_window(eval_month: str = EVAL_MONTH, n: int = FIT_WINDOW_MONTHS) -> list[str]:
    """The trailing `n` months before the evaluation month (D2): 2026-09 -> 2026-03..2026-08."""
    p = pd.Period(eval_month, "M")
    return [str(p - n + i) for i in range(n)]


def september_files(data_root: Path = None) -> list[str]:
    """Any arrival file of the evaluation month already on disk (the freeze stage refuses if there is one)."""
    data_root = Path(data_root or ROOT / "data")
    found = []
    for d in (data_root / "bts" / "parquet_arr", data_root / "cache" / "arrival"):
        if d.exists():
            found += [str(p) for p in sorted(d.glob(f"*{EVAL_MONTH}*"))]
    return found


def retest_files(data_root: Path = None) -> dict:
    """The data files the check stage hashes and the score stage re-hashes."""
    data_root = Path(data_root or ROOT / "data")
    return {"parquet_v2": data_root / "bts" / "parquet_v2" / f"{EVAL_MONTH}.parquet",
            "parquet_arr": data_root / "bts" / "parquet_arr" / f"{EVAL_MONTH}.parquet",
            "cache": data_root / "cache" / "arrival" / f"{EVAL_MONTH}.parquet",
            "pairs": data_root / "cache" / "arrival" / f"pairs_{EVAL_MONTH}.parquet"}


def model_paths(models_dir: Path = None) -> dict:
    models_dir = Path(models_dir or ROOT / "models")
    return {k: models_dir / v for k, v in MODEL_FILES.items()}


def hash_models(models_dir: Path = None) -> dict:
    out = {}
    for k, p in model_paths(models_dir).items():
        if not p.exists():
            raise SystemExit(f"{p} is missing (the frozen {k} file must be on disk)")
        out[k] = {"file": p.name, "sha256": sha256_file(p)}
    return out


# ------------------------------------------------------------------ the one fitted quantity: the level offset

def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_offset(p, y) -> float:
    """Unweighted maximum-likelihood a in logit(p') = logit(p) + a (slope fixed at 1), p clipped to [1e-6, 1-1e-6].
    The score equation sum(y - sigmoid(z + a)) = 0 is monotone in a, so it is solved by root finding."""
    from scipy.optimize import brentq
    z = _logit(p)
    y = np.asarray(y, dtype=float)
    if y.min() == y.max():
        raise ValueError("labels are all equal: the offset has no finite maximum-likelihood value")
    g = lambda a: float(y.sum() - (1.0 / (1.0 + np.exp(-(z + a)))).sum())      # noqa: E731
    return float(brentq(g, -15.0, 15.0, xtol=1e-12))


def apply_offsets(P, offsets) -> np.ndarray:
    """Arm L-S from arm L ladder rows (n, 11): add a_t on the logit scale per threshold, then rearrange."""
    P = np.asarray(P, dtype=float)
    out = np.empty_like(P)
    for k, t in enumerate(L.ARR_THRESHOLDS):
        out[:, k] = L.apply_logistic_recal(P[:, k], float(offsets[str(t)]), 1.0)
    return L.rearrange(out)


def collect_fit(store, months):
    """Arm-L ladder rows and arrival delays (flown flights, diverted = +inf) over the fit months."""
    P, y = [], []
    for ym in months:
        m, p = store.meta(ym), store.pred(ym)
        ok = E.flown_arrival_rows(m)
        P.append(p["arr"][ok])
        y.append(m["arr_delay"].to_numpy(float)[ok])
    return np.concatenate(P), np.concatenate(y)


def fit_offsets(store, months) -> dict:
    """a_t for every arrival threshold, fitted on `months` only."""
    if EVAL_MONTH in months or max(months) >= EVAL_MONTH:
        raise ValueError("the fit window must end before the evaluation month")
    P, y = collect_fit(store, months)
    out = {}
    for k, t in enumerate(L.ARR_THRESHOLDS):
        lab = (y >= t).astype(float)
        a = fit_offset(P[:, k], lab)
        out[str(t)] = {"a": a, "n": int(len(lab)), "fit_window_late_rate": float(lab.mean()),
                       "mean_pred_before": float(P[:, k].mean()),
                       "mean_pred_after": float(L.apply_logistic_recal(P[:, k].astype(float), a, 1.0).mean())}
    return out


def bl_table_to_json(tab) -> dict:
    return {"cell": [[h, int(c), float(v)] for (h, c), v in tab["cell"].items()],
            "bucket": [[g, int(c), float(v)] for (g, c), v in tab["bucket"].items()], "all": float(tab["all"])}


def bl_table_from_json(d) -> dict:
    return {"cell": {(h, int(c)): v for h, c, v in d["cell"]}, "bucket": {(g, int(c)): v for g, c, v in d["bucket"]},
            "all": float(d["all"])}


def climatology_ladder(store) -> dict:
    y = np.concatenate([store.meta(ym)["arr_delay"].to_numpy(float)[E.flown_arrival_rows(store.meta(ym))]
                        for ym in E.months_between(*CLIMATOLOGY_MONTHS)])
    return {str(t): float((y >= t).mean()) for t in L.ARR_THRESHOLDS}


def build_bl_table() -> dict:
    """Baseline B-L from the 2025-07..12 proxy pairs where both legs operated (clarifications 7 and 15)."""
    parts = []
    for ym in E.months_between(*BL_MONTHS):
        p = E.load_pairs(ym, 1.0)
        p = p[~p["c1"].to_numpy(bool) & ~p["c2"].to_numpy(bool)]
        parts.append(p[["hub", "layover", "miss"]])
    p = pd.concat(parts, ignore_index=True)
    return E.base_rate_table({"hub": p["hub"].to_numpy(), "layover": p["layover"].to_numpy().astype(int),
                              "miss": p["miss"].to_numpy(bool)})


# ------------------------------------------------------------------ stores

def make_store():
    """The same store as eval_arrival_connections.run: the ladder bundle and the production model."""
    artifact = ROOT / "models" / "arrival_ladder_v1.joblib"
    manifest = json.loads((ROOT / "models" / "arrival_ladder_v1.sha256.json").read_text())
    sha = sha256_file(artifact)
    if sha != manifest["sha256"]:
        raise SystemExit("models/arrival_ladder_v1.joblib does not match models/arrival_ladder_v1.sha256.json")
    bundle = L.load_bundle(artifact)
    import main as app
    return RetestStore(bundle, list(bundle.feature_names), app.model_service.model, sha)


class RetestStore(E.Store):
    """E.Store plus the evaluation month, which E.assert_window refuses. Months before it go through the parent."""

    def meta(self, ym):
        if ym != EVAL_MONTH:
            return super().meta(ym)
        refuse_if_unlock_set()
        if ym not in self._meta:
            self._meta[ym] = ad.load_cache(ym, retest=True)[E.CACHE_COLS]
        return self._meta[ym]

    def pred(self, ym):
        if ym != EVAL_MONTH:
            return super().pred(ym)
        refuse_if_unlock_set()
        if ym in self._pred:
            return self._pred[ym]
        path = ad.CACHE_DIR / f"pred_{ym}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            if str(z["sha"]) == self.sha:
                self._pred[ym] = {k: z[k] for k in ("arr", "dep", "cancel", "p_prod", "rearr")}
                return self._pred[ym]
        c = ad.load_cache(ym, retest=True)
        X = c[self.names].to_numpy(dtype=np.float32)
        raw_arr = np.stack([m.predict(X) for m in self.bundle.arr], axis=1)
        raw_dep = np.stack([m.predict(X) for m in self.bundle.dep], axis=1)
        d = {"arr": L.rearrange(raw_arr).astype(np.float32), "dep": L.rearrange(raw_dep).astype(np.float32),
             "cancel": self.bundle.cancel.predict(X).astype(np.float32),
             "p_prod": self.prod.predict_proba(c[self.names])[:, 1].astype(np.float32),
             "rearr": L.needs_rearrangement(raw_arr)}
        np.savez(path, sha=self.sha, **d)
        self._pred[ym] = d
        return d


# ------------------------------------------------------------------ freeze

def clean_tree() -> bool:
    return not _git("status", "--porcelain", "--untracked-files=no").stdout.strip()


def stage_freeze(freeze_path: Path = None, data_root: Path = None, store=None) -> dict:
    freeze_path = Path(freeze_path or FREEZE_PATH)
    refuse_if_unlock_set()
    if freeze_path.exists():
        raise SystemExit(f"freeze refused: {freeze_path.name} exists (the freeze is written once)")
    found = september_files(data_root)
    if found:
        raise SystemExit(f"freeze refused: {EVAL_MONTH} arrival files already exist, so a_t could have been "
                         f"influenced by September data: {found}")
    if not clean_tree():
        raise SystemExit("freeze refused: the working tree has uncommitted changes to tracked files")
    commit = _git("rev-parse", "HEAD").stdout.strip()
    for name, f in MODEL_FILES.items():              # the ladder is gitignored (kept on the Mac mini, pinned by hash)
        if name != "arrival_ladder" and _git("ls-files", "--error-unmatch", f"models/{f}").returncode:
            raise SystemExit(f"freeze refused: models/{f} is not tracked by git")
    models = hash_models()
    store = store or make_store()
    window = fit_window()
    log(f"fit window {window[0]}..{window[-1]}")
    offsets = fit_offsets(store, window)
    for t, o in offsets.items():
        log(f"t={t:>4}: a={o['a']:+.4f} (n={o['n']}, late rate {o['fit_window_late_rate']:.4f}, "
            f"mean p {o['mean_pred_before']:.4f} -> {o['mean_pred_after']:.4f})")
    freeze = {"protocol": PROTOCOL, "eval_month": EVAL_MONTH, "arm": "L-S",
              "fit_window": {"months": window, "rule": f"trailing {FIT_WINDOW_MONTHS} months before the eval month"},
              "offsets": offsets, "climatology_ladder": climatology_ladder(store),
              "climatology_months": list(CLIMATOLOGY_MONTHS),
              "b_l_table": bl_table_to_json(build_bl_table()), "b_l_months": list(BL_MONTHS),
              "code": {"commit": commit, "paths": CODE_PATHS}, "models": models, "created_utc": now_utc()}
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=1) + "\n")
    log(f"wrote {freeze_path.name}: freeze commit {commit[:10]}; commit this file before the check stage")
    return freeze


# ------------------------------------------------------------------ preflight shared by check and score

def load_freeze(path: Path = None) -> dict:
    path = Path(path or FREEZE_PATH)
    if not path.exists():
        raise SystemExit(f"{path.name} missing: run `--stage freeze` first")
    f = json.loads(path.read_text())
    if f.get("eval_month") != EVAL_MONTH:
        raise SystemExit(f"{path.name} is for {f.get('eval_month')}, not {EVAL_MONTH}")
    return f


def verify_models(freeze: dict, models_dir: Path = None) -> None:
    bad = []
    for name, m in freeze["models"].items():
        p = Path(models_dir or ROOT / "models") / m["file"]
        if not p.exists():
            bad.append(f"{name}: {p} missing")
        elif sha256_file(p) != m["sha256"]:
            bad.append(f"{name}: hash mismatch ({p})")
    if bad:
        raise SystemExit("frozen models do not match the freeze file:\n  " + "\n  ".join(bad))


def verify_frozen_code(freeze: dict, git=None) -> None:
    """A dirty tree is refused, and the frozen code must equal the freeze commit's (git blob/tree hashes, so later
    commits that only add the freeze file, the report or the results are fine)."""
    git = git or _git
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise SystemExit("refused: the working tree has uncommitted changes to tracked files")
    commit = freeze["code"]["commit"]
    if git("merge-base", "--is-ancestor", commit, "HEAD").returncode:
        raise SystemExit(f"refused: the freeze commit {commit[:10]} is not an ancestor of HEAD")
    bad = []
    for p in freeze["code"].get("paths", CODE_PATHS):
        a, b = git("rev-parse", f"HEAD:{p}"), git("rev-parse", f"{commit}:{p}")
        if a.returncode or b.returncode or a.stdout.strip() != b.stdout.strip():
            bad.append(p)
    if bad:
        raise SystemExit(f"refused: {bad} differ from the freeze commit {commit[:10]}; the code is frozen "
                         "(re-freeze needs a new amendment)")


def hash_data(data_root: Path = None) -> dict:
    return {k: sha256_file(p) for k, p in retest_files(data_root).items()}


def marker_path(results_dir: Path = None) -> Path:
    return Path(results_dir or RESULTS) / MARKER_NAME


def verify_check_marker(freeze_path: Path = None, results_dir: Path = None, data_root: Path = None) -> None:
    f = marker_path(results_dir)
    if not f.exists():
        raise SystemExit(f"score refused: no passing check marker ({f.name}); run `--stage check` until it passes")
    m = json.loads(f.read_text())
    if m.get("freeze_sha256") != sha256_file(Path(freeze_path or FREEZE_PATH)):
        raise SystemExit("score refused: the freeze file changed since `check` passed; repeat `--stage check`")
    try:
        files = hash_data(data_root)
    except FileNotFoundError as e:
        raise SystemExit(f"score refused: {e}")
    if m["files"] != files:
        raise SystemExit("score refused: the September files changed since `check` passed; repeat `--stage check`")


def preflight_score(force: bool = False, reason: str = None, results_dir: Path = None, freeze_path: Path = None,
                    data_root: Path = None, models_dir: Path = None, git=None) -> tuple[dict, Path]:
    """Everything `score` must satisfy before it touches any September data. Returns (freeze, result path).
    The original result is never overwritten: a re-run (only with --force-rerun and a --reason) writes
    arrival_retest_2026-09_rerun<N>.json next to it."""
    refuse_if_unlock_set()
    results_dir = Path(results_dir or RESULTS)
    freeze_path = Path(freeze_path or FREEZE_PATH)
    freeze = load_freeze(freeze_path)
    verify_models(freeze, models_dir)
    verify_frozen_code(freeze, git)
    verify_check_marker(freeze_path, results_dir, data_root)
    orig = results_dir / OUT_NAME
    attempts = results_dir / ATTEMPTS_NAME
    n = len(json.loads(attempts.read_text())) if attempts.exists() else 0
    if not (n or orig.exists()):
        if force:
            raise SystemExit("--force-rerun refused: there is no earlier attempt to re-run")
        return freeze, orig
    if not force:
        raise SystemExit(f"{attempts.name if n else orig.name} exists: the re-test is scored once (an attempt is "
                         "recorded before any outcome is read). --force-rerun --reason exists only for a bug fix")
    if not (reason or "").strip():
        raise SystemExit("--force-rerun needs --reason \"<why>\" (it is recorded in the result)")
    k = max(n, 1)
    while (results_dir / f"arrival_retest_{EVAL_MONTH}_rerun{k}.json").exists():
        k += 1
    log(f"FORCE-RERUN {k}: {reason} (the original result is kept; both are reported)")
    return freeze, results_dir / f"arrival_retest_{EVAL_MONTH}_rerun{k}.json"


def record_attempt(out: Path, freeze: dict, reason: str = None, results_dir: Path = None, git=None) -> None:
    """Written BEFORE any September outcome is loaded, so an interrupted run still counts as the one attempt."""
    git = git or _git
    f = Path(results_dir or RESULTS) / ATTEMPTS_NAME
    rec = json.loads(f.read_text()) if f.exists() else []
    rec.append({"started_utc": now_utc(), "output": out.name, "code_commit": git("rev-parse", "HEAD").stdout.strip(),
                "freeze_commit": freeze["code"]["commit"], "reason": reason,
                "models_sha256": {k: v["sha256"] for k, v in freeze["models"].items()}})
    tmp = f.with_suffix(".part")
    tmp.write_text(json.dumps(rec, indent=2) + "\n")
    tmp.rename(f)


# ------------------------------------------------------------------ check (no outcome-based number is printed)

def stage_check(freeze_path: Path = None, results_dir: Path = None, data_root: Path = None) -> bool:
    """Integrity checks on the September files. Prints PASS/FAIL lines and counts only."""
    import pyarrow.parquet as pq
    refuse_if_unlock_set()
    freeze_path = Path(freeze_path or FREEZE_PATH)
    files = retest_files(data_root)
    results = []

    def report(name, ok, detail=""):
        results.append(bool(ok))
        log(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))

    try:
        freeze = load_freeze(freeze_path)
        verify_models(freeze)
        report("freeze file present and frozen models match it", True)
        verify_frozen_code(freeze)
        report("working tree clean and frozen code equals the freeze commit", True)
    except SystemExit as e:
        report("freeze, models and frozen code", False, str(e))
    missing = [k for k, p in files.items() if not p.exists()]
    report("September files present (parquet_v2, parquet_arr, cache, pairs)", not missing, ", ".join(missing))
    if not missing:
        v2, arr = pd.read_parquet(files["parquet_v2"]), pd.read_parquet(files["parquet_arr"])
        d = pd.to_datetime(arr["FlightDate"])
        report("30 flight dates", d.dt.normalize().nunique() == 30, str(d.dt.normalize().nunique()))
        lo, hi = pd.Timestamp(f"{EVAL_MONTH}-01"), pd.Timestamp(f"{EVAL_MONTH}-30")
        report("no date outside September", bool(d.min() >= lo and d.max() <= hi))
        report(f"row count {EXPECTED_ROWS:,} (parquet_v2 and parquet_arr)", len(v2) == EXPECTED_ROWS and len(arr) == EXPECTED_ROWS,
               f"{len(v2):,} / {len(arr):,}")
        shared = [c for c in v2.columns if c in arr.columns]
        report("shared columns equal parquet_v2/2026-09", bool(arr[shared].equals(v2[shared])),
               f"{len(shared)} columns")
        cache = pq.ParquetFile(files["cache"])
        cm = pd.read_parquet(files["cache"], columns=["ym", "flight_date"])
        report("cache rows and months", cache.metadata.num_rows == EXPECTED_ROWS and set(cm["ym"]) == {EVAL_MONTH},
               f"{cache.metadata.num_rows:,}")
        cd = pd.to_datetime(cm["flight_date"])
        report("cache dates inside September", bool(cd.min() >= lo and cd.max() <= hi))
        pm = pd.read_parquet(files["pairs"], columns=["ym", "ym2", "arr_date"])
        report("pairs built, all in September", len(pm) > 0 and set(pm["ym"]) == {EVAL_MONTH}
               and set(pm["ym2"]) <= {EVAL_MONTH}, f"{len(pm):,} pairs")
        pdts = pd.to_datetime(pm["arr_date"])
        report("pair dates inside September", bool(pdts.min() >= lo and pdts.max() <= hi))
        del v2, arr
    ok = all(results)
    mf = marker_path(results_dir)
    if ok:
        mf.write_text(json.dumps({"month": EVAL_MONTH, "files": hash_data(data_root),
                                  "freeze_sha256": sha256_file(freeze_path), "created_utc": now_utc()}, indent=1) + "\n")
    elif mf.exists():
        mf.unlink()                                  # a failed check invalidates an earlier pass
    log("ALL CHECKS PASSED" if ok else "CHECKS FAILED: fix the data and repeat (no score has been computed)")
    return ok


# ------------------------------------------------------------------ decisions (thresholds unchanged)

def decide_arrival(rep: dict, d_prod: dict, d_clim: dict, cov: dict) -> dict:
    """A1-A4 and the arrival outcome. Copied statement for statement from `arrival_evaluation`
    (eval_arrival_connections.py); with one evaluation month the 'monthly' ECE is that month's."""
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
    return {"A1": {"pass": bool(a1), "detail": A1}, "A2": {"pass": bool(a2), "detail": A2},
            "A3": {"pass": bool(a3), "detail": A3}, "A4": {"pass": bool(a4), "detail": A4}, "outcome": outcome}


def arrival_passes(arrival_outcome: str) -> bool:
    """'A passes' for D4: the arrival outcome lets the ladder be used (clarification 16, re-test outcome)."""
    return arrival_outcome in E.LADDER_USABLE_FOR_CONNECTIONS


def confirmation_outcome(dec: dict) -> str:
    """Take-away-only C confirmation (D4), on the C decision block of E.connection_metrics with C4 re-read as a point
    estimate: C1-C3 failing means no connection numbers, C6 failing means none either; otherwise C4 point estimate
    below B-L confirms the copula, and a point estimate at or above 0 hands over to the original C4 rule (layover
    table B-L). C5 only decides whether the trip-level number is shown."""
    if not (dec["C1"]["pass"] and dec["C2"]["pass"] and dec["C3"]["pass"]):
        return "no_connection_numbers"
    if not dec["C6"]["pass"]:
        return "no_connection_numbers_c6_failed"
    if dec["C4"]["pass"]:
        return E.COPULA_SHIP
    return "serve_layover_table_B-L_itin-layover-table-v1"


def final_connection_outcome(arrival_outcome: str, c_outcome: str = None) -> str:
    """Clarification 16 as amended: 'arrival outcome' is the re-test outcome. The copula P(miss) is served only if
    the ladder may be used; the layover-table fallback and 'no numbers' are left as the C rules decide. If arrival
    failed, the C confirmation is not run and the copula is withheld."""
    if c_outcome is None:
        return "no_connection_numbers_arrival_ladder_not_shipped"
    if arrival_passes(arrival_outcome) or c_outcome != E.COPULA_SHIP:
        return c_outcome
    return "no_connection_numbers_arrival_ladder_not_shipped"


# ------------------------------------------------------------------ scoring

def arrival_scoring(store, freeze: dict, log=log) -> dict:
    E.TAIL_ARR = store.bundle.arr_tail
    W = E.collect_window(store, [EVAL_MONTH])
    log(f"arrival rows {len(W['y'])}")
    climo = [freeze["climatology_ladder"][str(t)] for t in L.ARR_THRESHOLDS]
    offsets = {t: o["a"] for t, o in freeze["offsets"].items()}
    P_L = W["P"].astype(float)
    P_LS = apply_offsets(P_L, offsets)
    k15 = list(L.ARR_THRESHOLDS).index(15)
    y15 = (W["y"] >= 15).astype(float)
    sq = lambda P: (P[:, k15] - y15) ** 2                                              # noqa: E731
    reports = {}
    for name, P in (("L-S", P_LS), ("L", P_L)):
        reports[name], rps, rps_c = E.arm_report(W, P, climo)
    base_prod = (W["p_prod"].astype(float) - y15) ** 2
    base_clim = (climo[k15] - y15) ** 2
    d_prod = E.day_block_ci(W["day"], sq(P_LS) - base_prod)
    d_clim = E.day_block_ci(W["day"], sq(P_LS) - base_clim)
    cov = E.coverage(P_LS, W["y"])
    log(f"coverage {cov}")
    decisions = decide_arrival(reports["L-S"], d_prod, d_clim, cov)
    return {"served_arm": "L-S", "offsets": offsets, "climatology_ladder": freeze["climatology_ladder"],
            "arms": reports, "decisions": decisions,
            "coverage_arm_L_reference": E.coverage(P_L, W["y"]),
            "baselines_t15": {"production_dep_as_arrival": E.binary_metrics(y15, W["p_prod"].astype(float)),
                              "climatology_brier": float(np.mean(base_clim))}}


def connection_confirmation(store, freeze: dict, log=log) -> dict:
    """September C confirmation with the validated model: R2+iso from models/connection_dependence_v1.json and arm L
    marginals (clarification 13). Nothing is fitted; B-L comes from the freeze file."""
    import arrival_service as svc
    d = json.loads((ROOT / "models" / MODEL_FILES["connection_dependence"]).read_text())
    dm = svc._dependence_from(d)
    recal = svc.load_recalibration(d)
    models = {dm.variant: dm, "R0": DependenceModel("R0", 0.0)}
    ev = E.score_month(store, EVAL_MONTH, CONFIG, models)
    p = np.asarray(ev["p"][dm.variant], dtype=float)
    pm = np.interp(p, recal[0], recal[1]) if recal is not None else p      # IsotonicRegression(out_of_bounds="clip")
    p_i = ev["p"]["R0"].astype(float)
    p_bl = E.bl_lookup(bl_table_from_json(freeze["b_l_table"]), ev["hub"], ev["layover"])
    res = E.connection_metrics(ev, pm, p_i, p_bl)
    dec = res["decisions"]
    d_bl = dec["C4"]["detail"]["brier_diff_vs_B-L"]
    dec["C4"] = {"pass": bool(d_bl["point"] < 0), "detail": {"brier_diff_vs_B-L": d_bl,
                                                          "rule": "point estimate below B-L; CI reported (D4)",
                                                          "ci_entirely_below_0": bool(d_bl["hi"] < 0)}}
    dec["original_outcome_with_ci_rule"] = dec.pop("outcome")
    dec["outcome"] = confirmation_outcome(dec)
    log(f"connection confirmation outcome {dec['outcome']}")
    res["chosen_variant"] = d.get("chosen")
    return res


def stage_score(force: bool = False, reason: str = None) -> None:
    freeze, out = preflight_score(force, reason)
    record_attempt(out, freeze, reason)                       # before any September outcome is loaded
    store = make_store()
    if store.sha != freeze["models"]["arrival_ladder"]["sha256"]:
        raise SystemExit("score refused: the loaded ladder is not the frozen one")
    t0 = time.time()
    arrival = arrival_scoring(store, freeze)
    a_out = arrival["decisions"]["outcome"]
    if arrival_passes(a_out):
        conn = connection_confirmation(store, freeze)
        c_out = conn["decisions"]["outcome"]
    else:
        conn = {"run": False, "reason": f"arrival re-test outcome {a_out}: the C confirmation is not run (D4)"}
        c_out = None
    final = final_connection_outcome(a_out, c_out)
    result = {"protocol": PROTOCOL, "eval_month": EVAL_MONTH, "run_utc": now_utc(),
              "force_rerun": {"reason": reason, "output": out.name} if force else None,
              "code": {"commit": _git("rev-parse", "HEAD").stdout.strip(), "freeze_commit": freeze["code"]["commit"]},
              "freeze_sha256": sha256_file(FREEZE_PATH), "models": freeze["models"], "arrival": arrival,
              "connections": conn, "final_connection_outcome": final, "seconds": round(time.time() - t0),
              "d6_note": "If A failed there are no more attempts on 2026 data; the next needs a new protocol on "
                         "2027-01+ (D6). D5: a take-away-only October confirmation in December, not implemented here."}
    out.write_text(json.dumps(E.jsonable(result), indent=1) + "\n")
    log("wrote", out)
    log("arrival outcome:", a_out, "| connection final outcome:", final)
    log("now check that the unlock is not set: env | grep FLYRELY_HOLDOUT || echo OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["freeze", "check", "score"], required=True)
    ap.add_argument("--force-rerun", action="store_true", help="score only; writes arrival_retest_2026-09_rerun<N>.json")
    ap.add_argument("--reason", help="score --force-rerun: why (recorded in the result)")
    a = ap.parse_args(argv)
    if (a.force_rerun or a.reason) and a.stage != "score":
        raise SystemExit("--force-rerun/--reason apply to --stage score only")
    if a.reason and not a.force_rerun:
        raise SystemExit("--reason is for --force-rerun")
    RESULTS.mkdir(exist_ok=True)
    if a.stage == "freeze":
        stage_freeze()
    elif a.stage == "check":
        sys.exit(0 if stage_check() else 1)
    else:
        stage_score(a.force_rerun, a.reason)


if __name__ == "__main__":
    main()
