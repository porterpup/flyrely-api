"""
Exceedance ladders: calibrated S(t) = P(delay >= t | flight operates) at fixed thresholds, interpolated into a
full delay distribution (protocol: backtest/PROTOCOL_ARRIVAL_CONNECTIONS.md, section 2.1 of the plan).

A diverted flight counts as +infinity (exceeds every threshold). Below the lowest and above the highest threshold the
curve follows fit-window empirical tail tables (one per ladder), capped at CAP minutes. Nothing here imports the
backtest code, so it can be used by the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

ARR_THRESHOLDS = (-30, -15, 0, 15, 30, 45, 60, 90, 120, 180, 300)
DEP_THRESHOLDS = (-5, 0, 15, 30, 45, 60, 90, 120, 180, 300)
GRID_LO, CAP = -60, 600
GRID = np.arange(GRID_LO, CAP + 1)
SEVERITY_POINTS = (15, 45, 120)
VERSION = "arr-ladder-v1"


# ---------------------------------------------------------------- tails

@dataclass(frozen=True)
class TailTable:
    """lo_cdf[i] = P(delay < x_i | delay < t_min) on x = GRID_LO..t_min; hi_sf[i] = P(delay >= x_i | delay >= t_max)
    on x = t_max..CAP (a diverted flight counts in every x)."""
    t_min: int
    t_max: int
    lo_cdf: np.ndarray
    hi_sf: np.ndarray


def build_tail_table(delay: np.ndarray, t_min: int, t_max: int) -> TailTable:
    """From fit-window delays of flown flights (np.inf for diverted). No NaNs."""
    d = np.asarray(delay, dtype=float)
    lo = d[d < t_min]
    hi = d[d >= t_max]
    xs_lo = np.arange(GRID_LO, t_min + 1)
    xs_hi = np.arange(t_max, CAP + 1)
    if len(lo):
        s = np.sort(lo)
        lo_cdf = np.searchsorted(s, xs_lo, side="left") / len(s)
    else:
        lo_cdf = np.where(xs_lo >= t_min, 1.0, 0.0)
    lo_cdf = np.maximum.accumulate(lo_cdf)
    lo_cdf[-1] = 1.0
    if len(hi):
        s = np.sort(hi)
        hi_sf = 1.0 - np.searchsorted(s, xs_hi, side="left") / len(s)
    else:
        hi_sf = np.where(xs_hi <= t_max, 1.0, 0.0)
    hi_sf = np.minimum.accumulate(hi_sf)
    hi_sf[0] = 1.0
    return TailTable(int(t_min), int(t_max), lo_cdf.astype(np.float64), hi_sf.astype(np.float64))


# ---------------------------------------------------------------- curves

def rearrange(P: np.ndarray) -> np.ndarray:
    """Force S to be non-increasing in the threshold (sorting across thresholds), inside [0, 1]."""
    return np.clip(-np.sort(-np.asarray(P, dtype=float), axis=1), 0.0, 1.0)


def needs_rearrangement(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    return (np.diff(P, axis=1) > 1e-12).any(axis=1)


def curve_matrix(P: np.ndarray, thresholds, tail: TailTable) -> np.ndarray:
    """S on the 1-minute GRID for each row of P (n, k), P already rearranged. Exact at the thresholds."""
    P = np.asarray(P, dtype=float)
    th = np.asarray(thresholds, dtype=float)
    out = np.empty((P.shape[0], len(GRID)))
    below = GRID < th[0]
    above = GRID > th[-1]
    inside = ~(below | above)
    if below.any():
        out[:, below] = 1.0 - (1.0 - P[:, :1]) * tail.lo_cdf[GRID[below] - GRID_LO][None, :]
    if above.any():
        out[:, above] = P[:, -1:] * tail.hi_sf[GRID[above] - tail.t_max][None, :]
    g = GRID[inside].astype(float)
    j = np.clip(np.searchsorted(th, g, side="right") - 1, 0, len(th) - 2)
    w = (g - th[j]) / (th[j + 1] - th[j])
    out[:, inside] = P[:, j] * (1.0 - w)[None, :] + P[:, j + 1] * w[None, :]
    return np.clip(out, 0.0, 1.0)


def quantile_matrix(S: np.ndarray, q: float) -> np.ndarray:
    """Delay (minutes) x with P(delay <= x) = q, i.e. S(x) = 1 - q, linearly interpolated on GRID; one per row."""
    target = 1.0 - q
    hit = S <= target
    idx = np.where(hit.any(axis=1), hit.argmax(axis=1), len(GRID) - 1)
    prev = np.clip(idx - 1, 0, None)
    rows = np.arange(S.shape[0])
    s0, s1 = S[rows, prev], S[rows, idx]
    frac = np.where(s0 > s1, (s0 - target) / np.where(s0 > s1, s0 - s1, 1.0), 0.0)
    x = GRID[prev] + np.clip(frac, 0, 1) * (GRID[idx] - GRID[prev])
    x = np.where(idx == 0, GRID[0], x)
    return np.where(hit.any(axis=1), x, float(CAP))


def severity_matrix(P: np.ndarray, thresholds) -> np.ndarray:
    """(n, 3): minor, moderate, severe given late (>= 15). Rows with S15 == 0 give 1, 0, 0."""
    th = list(thresholds)
    s15, s45, s120 = (np.asarray(P)[:, th.index(t)] for t in SEVERITY_POINTS)
    safe = np.where(s15 > 0, s15, 1.0)
    out = np.stack([(s15 - s45) / safe, (s45 - s120) / safe, s120 / safe], axis=1)
    out[s15 <= 0] = (1.0, 0.0, 0.0)
    return out


@dataclass(frozen=True)
class Ladder:
    """One flight's distribution: thresholds (minutes, ascending) and S = P(delay >= t | operates), non-increasing."""
    thresholds: tuple
    probs: tuple
    tail: TailTable

    @classmethod
    def from_probs(cls, thresholds, probs, tail: TailTable) -> "Ladder":
        return cls(tuple(thresholds), tuple(rearrange(np.asarray(probs, dtype=float)[None, :])[0]), tail)

    def _curve(self) -> np.ndarray:
        return curve_matrix(np.asarray(self.probs)[None, :], self.thresholds, self.tail)

    def sf(self, x) -> np.ndarray:
        """P(delay >= x); x in minutes (clamped to the grid -60..600)."""
        return np.interp(np.asarray(x, dtype=float), GRID, self._curve()[0])

    def quantile(self, q: float) -> float:
        return float(quantile_matrix(self._curve(), q)[0])

    def severity(self) -> dict:
        m, mo, s = severity_matrix(np.asarray(self.probs)[None, :], self.thresholds)[0]
        return {"minor": float(m), "moderate": float(mo), "severe": float(s)}

    def prob_late(self, t: int = 15) -> float:
        return float(self.probs[list(self.thresholds).index(t)])


# ---------------------------------------------------------------- recalibration arm (L-R)

def fit_logistic_recal(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Intercept a and slope b of logit(p') = a + b * logit(p) (no penalty)."""
    from sklearn.linear_model import LogisticRegression
    z = _logit(p)[:, None]
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=200).fit(z, y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def apply_logistic_recal(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a + b * _logit(p))))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- the artifact

@dataclass
class ThresholdModel:
    threshold: Optional[int]
    model: object
    iso: object

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.iso.predict(self.model.predict_proba(X)[:, 1]), dtype=float)


@dataclass
class LadderBundle:
    version: str
    feature_names: list
    arr: list
    dep: list
    cancel: ThresholdModel
    arr_tail: TailTable
    dep_tail: TailTable
    data_through: str
    recal: Optional[dict] = field(default=None)       # {"arr": [(a, b), ...], "dep": [...]} when arm L-R is served

    def predict_arrays(self, X) -> dict:
        """Rearranged S matrices and cancel probability for a feature matrix with the 18 columns in order."""
        arr = rearrange(np.stack([m.predict(X) for m in self.arr], axis=1))
        dep = rearrange(np.stack([m.predict(X) for m in self.dep], axis=1))
        return {"arr": arr, "dep": dep, "cancel": self.cancel.predict(X)}

    def ladders(self, X) -> list[tuple[Ladder, Ladder, float]]:
        r = self.predict_arrays(X)
        return [(Ladder(tuple(ARR_THRESHOLDS), tuple(a), self.arr_tail), Ladder(tuple(DEP_THRESHOLDS), tuple(d), self.dep_tail), float(c))
                for a, d, c in zip(r["arr"], r["dep"], r["cancel"])]

    def save(self, path) -> None:
        joblib.dump(self, path, compress=3)


def load_bundle(path, feature_names_path=None) -> LadderBundle:
    b = joblib.load(path)
    if feature_names_path is None:
        feature_names_path = Path(__file__).resolve().parent.parent / "models" / "feature_names.joblib"
    expected = list(joblib.load(feature_names_path))
    if list(b.feature_names) != expected:
        raise ValueError("ladder feature names differ from models/feature_names.joblib")
    return b
