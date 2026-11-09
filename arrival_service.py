"""Serving side of the arrival ladder and the connection dependence model (T6).

The ladder artifact (models/arrival_ladder_v1.joblib) is gitignored and kept on the Mac (decision D4); its sha256 is in
models/arrival_ladder_v1.sha256.json. A file whose hash differs from the manifest is refused. Path overrides:
FLYRELY_ARRIVAL_LADDER_PATH, FLYRELY_ARRIVAL_LADDER_MANIFEST, FLYRELY_CONNECTION_DEPENDENCE_PATH.

Nothing here invents a number: when the ladder (or, for itineraries, the dependence file) is missing, callers get
`None` / a PredictionUnavailable and say so.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from flyrely_dist.connections import DependenceModel, LegForecast
from flyrely_dist.ladder import LadderBundle, load_bundle
from flyrely_dist.providers import BookingLadderProvider, LegSpec

logger = logging.getLogger("flyrely.arrival")
MODEL_DIR = Path(__file__).resolve().parent / "models"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# The only T5 connection outcome under which copula miss probabilities may be served (PROTOCOL_ARRIVAL_CONNECTIONS.md;
# backtest/eval_arrival_connections.py). Any other recorded outcome (the C4 layover-table fallback, or no connection
# numbers at all) makes the itinerary endpoint refuse with 503 instead of serving copula numbers.
SHIP_OUTCOME = "ship_miss_probability_for_validated_connections"


def load_dependence(path) -> DependenceModel:
    return _dependence_from(json.loads(Path(path).read_text()))


def _dependence_from(d: dict) -> DependenceModel:
    cells = {k: (float(v[0]), int(v[1])) for k, v in (d.get("by_cell") or {}).items()}
    return DependenceModel(str(d.get("variant", "R1")), float(d.get("global_rho", 0.0)), cells,
                           float(d.get("shrink_n", 5000.0)))


def load_recalibration(d: dict) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Isotonic breakpoints written by the T5 eval ({"type": "isotonic", "x": [...], "y": [...]}, from sklearn's
    IsotonicRegression(out_of_bounds="clip") X_thresholds_ / y_thresholds_), or None. Raises on a malformed block."""
    r = d.get("recalibration")
    if r is None:
        return None
    if r.get("type") != "isotonic":
        raise ValueError(f"unknown recalibration type {r.get('type')!r}")
    x, y = np.asarray(r["x"], dtype=float), np.asarray(r["y"], dtype=float)
    if x.ndim != 1 or len(x) == 0 or x.shape != y.shape or np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0) \
            or not np.all(np.isfinite(x)) or not np.all((y >= 0) & (y <= 1)):
        raise ValueError("malformed isotonic recalibration")
    return x, y


def apply_recalibration(p: float, recal: Optional[tuple[np.ndarray, np.ndarray]]) -> float:
    """Same as IsotonicRegression(out_of_bounds="clip").predict: linear between breakpoints, flat beyond the ends."""
    if recal is None:
        return float(p)
    x, y = recal
    return float(np.interp(p, x, y))


class ArrivalService:
    def __init__(self, ladder_path=None, manifest_path=None, dependence_path=None,
                 feature_fn: Optional[Callable] = None, enabled: bool = True):
        self.ladder_path = Path(ladder_path or os.getenv("FLYRELY_ARRIVAL_LADDER_PATH")
                                or MODEL_DIR / "arrival_ladder_v1.joblib")
        self.manifest_path = Path(manifest_path or os.getenv("FLYRELY_ARRIVAL_LADDER_MANIFEST")
                                  or MODEL_DIR / "arrival_ladder_v1.sha256.json")
        self.dependence_path = Path(dependence_path or os.getenv("FLYRELY_CONNECTION_DEPENDENCE_PATH")
                                    or MODEL_DIR / "connection_dependence_v1.json")
        self.feature_fn = feature_fn
        self.bundle: Optional[LadderBundle] = None
        self.provider: Optional[BookingLadderProvider] = None
        self.dependence: Optional[DependenceModel] = None
        self.recalibration: Optional[tuple] = None
        self.decision_outcome: Optional[str] = None
        self.load_error: Optional[str] = None
        self.dependence_error: Optional[str] = None
        if enabled:
            self._load()

    def _load(self) -> None:
        try:
            if not self.ladder_path.exists():
                raise FileNotFoundError(f"{self.ladder_path.name} not found")
            manifest = json.loads(self.manifest_path.read_text())
            if sha256_file(self.ladder_path) != manifest["sha256"]:
                raise ValueError("ladder file does not match its sha256 manifest")
            self.bundle = load_bundle(self.ladder_path)
            self.provider = BookingLadderProvider(self.bundle, self.feature_fn)
            logger.info(f"Arrival ladder loaded ({self.bundle.version}, data through {self.bundle.data_through})")
        except Exception as e:
            self.bundle = self.provider = None
            self.load_error = f"{type(e).__name__}: {e}"
            logger.error(f"Arrival ladder unavailable: {self.load_error}")
        try:
            d = json.loads(self.dependence_path.read_text())
            outcome = d.get("decision_outcome")
            if outcome is not None and outcome != SHIP_OUTCOME:
                raise ValueError(f"connection evaluation outcome is {outcome!r}; copula numbers are not served")
            self.recalibration = load_recalibration(d)
            self.dependence = _dependence_from(d)
            self.decision_outcome = outcome
        except Exception as e:
            self.dependence = self.recalibration = None
            self.dependence_error = f"{type(e).__name__}: {e}"
            logger.error(f"Connection dependence unavailable: {self.dependence_error}")

    @property
    def loaded(self) -> bool:
        return self.provider is not None

    @property
    def version(self) -> Optional[str]:
        return self.bundle.version if self.bundle else None

    def calibrate_miss(self, p: float) -> float:
        """Apply the T5 isotonic recalibration (if any) to a copula P(miss)."""
        return apply_recalibration(p, self.recalibration)

    def forecast(self, legs: list[LegSpec]) -> list[LegForecast]:
        return self.provider.forecast(legs)


def arrival_block(f: LegForecast, version: str, risk_fn: Callable[[float], str]) -> dict:
    """The `arrival` block of the /predict response (plan section 2.3). Percent-only copy is the app's job."""
    lad = f.arr
    p15 = lad.prob_late(15)
    sev = lad.severity()
    label = max(sev, key=sev.get)
    ranges = {"minor": "15–44 min", "moderate": "45–119 min", "severe": "120+ min"}
    return {
        "probability_late_15": round(p15, 3),
        "risk_level": risk_fn(p15),
        "severity": {"minor_pct": round(sev["minor"], 3), "moderate_pct": round(sev["moderate"], 3),
                     "severe_pct": round(sev["severe"], 3), "expected_delay_label": label,
                     "expected_delay_range": ranges[label]},
        "quantiles_minutes": {"p50": round(lad.quantile(0.5)), "p80": round(lad.quantile(0.8)),
                              "p90": round(lad.quantile(0.9))},
        "exceedance": [{"minutes": int(t), "probability": round(float(p), 4)}
                       for t, p in zip(lad.thresholds, lad.probs)],
        "cancel_probability": round(f.p_cancel, 4),
        "model_version": version,
    }
