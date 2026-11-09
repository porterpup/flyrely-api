"""
Missed-connection probability from two delay marginals and a Gaussian copula (plan section 2.2).

Leg 1 arrives at hub H, leg 2 departs H. X = leg 1 arrival delay, Y = leg 2 departure delay, s = layover - MCT.
Missed  <=>  X - Y > s  (given both legs operate); X - Y == s is NOT a miss. A diverted leg 1 (X = infinity) and any
leg 1 beyond the 600-minute cap count as a miss. Pure numpy; a marginal is anything with `.sf(x)` = P(delay >= x)
(a `Ladder`, or a test double).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr as _ndtr, ndtri as _ndtri

from .ladder import CAP, GRID, GRID_LO, Ladder, curve_matrix

_EPS = 1e-9


def _ppf(u):
    return _ndtri(np.clip(np.asarray(u, dtype=float), _EPS, 1 - _EPS))


@dataclass(frozen=True)
class LegForecast:
    arr: Ladder
    dep: Ladder
    p_cancel: float
    provider: str = "booking-ladder-v1"
    horizon: str = "booking"


def layover_bucket(layover_min: float) -> str:
    return "lt60" if layover_min < 60 else "60_120" if layover_min < 120 else "120_240"


@dataclass(frozen=True)
class DependenceModel:
    """rho by variant: R0 independence, R1 one global rho, R2 hub x layover bucket shrunk toward the global rho."""
    variant: str = "R0"
    global_rho: float = 0.0
    by_cell: dict = field(default_factory=dict)       # "HUB|bucket" -> (rho, n)
    shrink_n: float = 5000.0

    def rho(self, hub: str | None = None, layover_min: float = 60.0) -> float:
        if self.variant == "R0":
            return 0.0
        if self.variant == "R1" or hub is None:
            return float(self.global_rho)
        cell = self.by_cell.get(f"{hub}|{layover_bucket(layover_min)}")
        if not cell:
            return float(self.global_rho)
        r, n = cell
        w = n / (n + self.shrink_n)
        return float(w * r + (1 - w) * self.global_rho)


def miss_probability(f1: LegForecast, f2: LegForecast, layover_min: float, mct_min: float,
                     dep: DependenceModel | None = None, hub: str | None = None, rho: float | None = None) -> float:
    """P(miss | both legs operate)."""
    if rho is None:
        rho = (dep or DependenceModel()).rho(hub, layover_min)
    return _miss(f1.arr, f2.dep, layover_min - mct_min, rho)


def _miss(arr_marginal, dep_marginal, s: float, rho: float) -> float:
    x = GRID.astype(float)
    s1 = np.asarray(arr_marginal.sf(x), dtype=float)
    s1_next = np.append(s1[1:], s1[-1])
    pmf = np.clip(s1 - s1_next, 0.0, None)                   # P(X = x) for integer x in the grid
    tail_mass = float(s1[-1])                                 # >= CAP or diverted: a miss
    f2 = 1.0 - np.asarray(dep_marginal.sf(x - s), dtype=float)   # P(Y < x - s)
    if abs(rho) < 1e-12:
        cond = f2
    else:
        u1 = np.clip(1.0 - (s1 + s1_next) / 2.0, _EPS, 1 - _EPS)       # mid-point PIT of X
        z1 = _ppf(u1)
        z2 = _ppf(f2)
        cond = _ndtr((z2 - rho * z1) / np.sqrt(1.0 - rho * rho))
        cond = np.where(f2 <= _EPS, 0.0, np.where(f2 >= 1 - _EPS, 1.0, cond))
    return float(np.clip((pmf * cond).sum() + tail_mass, 0.0, 1.0))


def disruption_probability(p_cancel: list[float], misses: list[float]) -> float:
    """P(any cancellation or missed connection) = 1 - prod(1 - c_i) * prod(1 - miss_k), independence assumed
    (rule C5 tests that assumption for two-leg trips)."""
    ok = 1.0
    for c in p_cancel:
        ok *= 1.0 - c
    for m in misses:
        ok *= 1.0 - m
    return float(1.0 - ok)


def trip_summary(forecasts: list[LegForecast], connections: list[float | None]) -> dict:
    """`connections[k]` is the miss probability of connection k (None when not validated). Three or more legs
    multiply unconditional marginals, so the result is flagged approximate."""
    if len(connections) != len(forecasts) - 1:
        raise ValueError("need one connection per adjacent pair of legs")
    known = [m for m in connections if m is not None]
    complete = len(known) == len(connections)
    return {"disruption_probability": disruption_probability([f.p_cancel for f in forecasts], known)
            if complete else None,
            "approximate": len(forecasts) >= 3}


# ---------------------------------------------------------------- fitting rho from PIT values

def sf_at(P: np.ndarray, thresholds, tail, x: np.ndarray) -> np.ndarray:
    """S(x_i) for each row i of P (n, k) at that row's own x_i, exactly the Ladder curve."""
    P = np.asarray(P, dtype=float)
    th = np.asarray(thresholds, dtype=float)
    x = np.asarray(x, dtype=float)
    rows = np.arange(len(P))
    out = np.empty(len(x))
    lo, hi = x < th[0], x > th[-1]
    mid = ~(lo | hi)
    xc = np.clip(x, GRID_LO, CAP)
    if lo.any():
        out[lo] = 1.0 - (1.0 - P[rows[lo], 0]) * _interp(tail.lo_cdf, GRID_LO, xc[lo])
    if hi.any():
        out[hi] = P[rows[hi], -1] * _interp(tail.hi_sf, tail.t_max, np.maximum(xc[hi], tail.t_max))
    if mid.any():
        j = np.clip(np.searchsorted(th, x[mid], side="right") - 1, 0, len(th) - 2)
        w = (x[mid] - th[j]) / (th[j + 1] - th[j])
        out[mid] = P[rows[mid], j] * (1 - w) + P[rows[mid], j + 1] * w
    return out


def _interp(table: np.ndarray, x0: int, x: np.ndarray) -> np.ndarray:
    return np.interp(x, np.arange(x0, x0 + len(table)), table)


def pit_normal_scores(P, thresholds, tail, x, rng: np.random.Generator) -> np.ndarray:
    """Normal score of the PIT of integer-valued delays x under each row's ladder, randomised within the bin
    [F(x), F(x+1)) where F(x) = 1 - S(x)."""
    s0 = sf_at(P, thresholds, tail, x)
    s1 = sf_at(P, thresholds, tail, np.asarray(x, dtype=float) + 1.0)
    u = 1.0 - s0 + rng.random(len(s0)) * np.clip(s0 - s1, 0.0, None)
    return _ppf(u)


def fit_dependence(z1: np.ndarray, z2: np.ndarray, hubs: np.ndarray, layovers: np.ndarray, weights=None,
                   variant: str = "R2", shrink_n: float = 5000.0) -> DependenceModel:
    """rho = correlation of normal scores (optionally weighted) globally and per hub x layover bucket."""
    if variant == "R0":
        return DependenceModel("R0")
    g = _wcorr(z1, z2, weights)
    if variant == "R1":
        return DependenceModel("R1", g)
    buckets = np.array([layover_bucket(v) for v in layovers])
    cells = {}
    for hub in np.unique(hubs):
        for b in ("lt60", "60_120", "120_240"):
            m = (hubs == hub) & (buckets == b)
            if m.sum() >= 30:
                cells[f"{hub}|{b}"] = (_wcorr(z1[m], z2[m], None if weights is None else weights[m]), int(m.sum()))
    return DependenceModel("R2", g, cells, shrink_n)


def _wcorr(a, b, w=None) -> float:
    w = np.ones(len(a)) if w is None else np.asarray(w, dtype=float)
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    cov = (w * (a - ma) * (b - mb)).sum()
    return float(np.clip(cov / np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum()), -0.99, 0.99))
