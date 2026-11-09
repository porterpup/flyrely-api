"""
Train the arrival, departure and cancellation ladders (protocol: PROTOCOL_ARRIVAL_CONNECTIONS.md).

    OMP_NUM_THREADS=4 nice python backtest/train_ladders.py [--out models/arrival_ladder_v1.joblib] [--quick]

Fit window 2025-01..06 (model weights), calibration 2025-07..12 (isotonic), through backtest.fit_calibrated
(HGB, seed 42, unchanged hyperparameters). Reads the monthly caches only (arrival_data.py); nothing dated
2026 is read here: evaluation happens once, in eval_arrival_connections.py. Never runs with FLYRELY_HOLDOUT_UNLOCK.
Logs per-model time, artifact size and single-row latency; writes a sha256 manifest next to the artifact.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
import arrival_data as ad  # noqa: E402
from flyrely_dist import ladder as L  # noqa: E402

FIT_MONTHS = ("2025-01", "2025-06")
CAL_MONTHS = ("2025-07", "2025-12")
DEFAULT_OUT = ROOT / "models" / "arrival_ladder_v1.joblib"


def feature_names() -> list:
    return list(joblib.load(ROOT / "models" / "feature_names.joblib"))


def load_window(start: str, end: str, names: list, cache_dir=None):
    frames = [ad.load_cache(ym, cache_dir) for ym in ad.months(start, end)]
    df = pd.concat(frames, ignore_index=True)
    X = df[names].to_numpy(dtype=np.float32)
    lab = {"arr_delay": df["arr_delay"].to_numpy(dtype=np.float64), "dep_delay": df["dep_delay"].to_numpy(dtype=np.float64),
           "cancelled": df["cancelled"].to_numpy(dtype=bool)}
    return X, lab


def _rows(lab, which):
    m = ~lab["cancelled"] & ~np.isnan(lab[which])
    return m


def train_bundle(X_fit, lab_fit, X_cal, lab_cal, names, fitter, data_through="2025-12", log=print) -> L.LadderBundle:
    def fit_one(y_fit, y_cal, mask_fit, mask_cal, tag, threshold):
        t0 = time.time()
        model, iso = fitter(X_fit[mask_fit], y_fit, X_cal[mask_cal], y_cal)
        log(f"  {tag} done in {time.time() - t0:.0f}s (base rate fit {y_fit.mean():.4f})")
        return L.ThresholdModel(threshold, model, iso)

    ladders = {}
    for which, ths in (("arr_delay", L.ARR_THRESHOLDS), ("dep_delay", L.DEP_THRESHOLDS)):
        mf, mc = _rows(lab_fit, which), _rows(lab_cal, which)
        out = []
        for t in ths:
            out.append(fit_one((lab_fit[which][mf] >= t).astype(int), (lab_cal[which][mc] >= t).astype(int), mf, mc,
                               f"{which[:3]} >= {t}", t))
        ladders[which] = out
    allf, allc = np.ones(len(X_fit), dtype=bool), np.ones(len(X_cal), dtype=bool)
    cancel = fit_one(lab_fit["cancelled"].astype(int), lab_cal["cancelled"].astype(int), allf, allc, "cancel", None)
    tails = {}
    for which, ths in (("arr_delay", L.ARR_THRESHOLDS), ("dep_delay", L.DEP_THRESHOLDS)):
        d = lab_fit[which][_rows(lab_fit, which)]
        tails[which] = L.build_tail_table(d, ths[0], ths[-1])
    return L.LadderBundle(L.VERSION, list(names), ladders["arr_delay"], ladders["dep_delay"], cancel,
                          tails["arr_delay"], tails["dep_delay"], data_through)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path) -> Path:
    mp = path.with_suffix(".sha256.json")
    mp.write_text(json.dumps({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
                              "version": L.VERSION, "location": "kept on the Mac mini; gitignored (D4)"}, indent=2) + "\n")
    return mp


def single_row_latency_ms(bundle: L.LadderBundle, X, n: int = 20) -> float:
    row = X[:1]
    bundle.predict_arrays(row)                       # warm
    t0 = time.perf_counter()
    for _ in range(n):
        bundle.predict_arrays(row)
    return (time.perf_counter() - t0) / n * 1000


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--quick", action="store_true", help="2025-01..02 fit, 2025-03 calibration; writes <out>.quick")
    args = ap.parse_args(argv)
    ad.guard_month("2025-01")                         # also exits if the unlock variable is set
    import backtest as bt
    names = feature_names()
    fit_m, cal_m = (("2025-01", "2025-02"), ("2025-03", "2025-03")) if args.quick else (FIT_MONTHS, CAL_MONTHS)
    out = Path(args.out + (".quick" if args.quick else ""))
    t0 = time.time()
    X_fit, lab_fit = load_window(*fit_m, names)
    X_cal, lab_cal = load_window(*cal_m, names)
    print(f"loaded fit {X_fit.shape} cal {X_cal.shape} in {time.time() - t0:.0f}s", flush=True)
    bundle = train_bundle(X_fit, lab_fit, X_cal, lab_cal, names, bt.fit_calibrated, cal_m[1],
                          log=lambda s: print(s, flush=True))
    bundle.save(out)
    mp = write_manifest(out)
    print(f"artifact {out.name}: {out.stat().st_size / 1e6:.1f} MB, manifest {mp.name}")
    b2 = L.load_bundle(out)
    print(f"single-row latency, 22 models: {single_row_latency_ms(b2, X_cal):.1f} ms")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
