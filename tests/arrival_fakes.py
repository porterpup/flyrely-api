"""A synthetic ladder bundle (no data files) and an ArrivalService built from it, for the T6 API tests."""
import numpy as np

from arrival_service import ArrivalService
from flyrely_dist import ladder as L
from flyrely_dist.connections import DependenceModel
from flyrely_dist.providers import BookingLadderProvider

ARR_P = [0.95, 0.85, 0.6, 0.23, 0.16, 0.12, 0.09, 0.06, 0.04, 0.02, 0.008]
DEP_P = [0.6, 0.4, 0.2, 0.12, 0.09, 0.07, 0.045, 0.03, 0.015, 0.006]
CANCEL_P = 0.012
FEATURE_NAMES = ["a", "b"]


class _Const:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


class _Identity:
    def predict(self, p):
        return np.asarray(p)


def _tail(thresholds):
    rng = np.random.default_rng(0)
    d = np.where(rng.random(40000) < 0.25, rng.exponential(45, 40000) + 10, rng.normal(-8, 12, 40000))
    return L.build_tail_table(d, thresholds[0], thresholds[-1])


def make_bundle() -> L.LadderBundle:
    tm = lambda t, p: L.ThresholdModel(t, _Const(p), _Identity())
    return L.LadderBundle("arr-ladder-v1", FEATURE_NAMES,
                          [tm(t, p) for t, p in zip(L.ARR_THRESHOLDS, ARR_P)],
                          [tm(t, p) for t, p in zip(L.DEP_THRESHOLDS, DEP_P)],
                          tm(None, CANCEL_P), _tail(L.ARR_THRESHOLDS), _tail(L.DEP_THRESHOLDS), "2025-12")


def make_service(airlines_seen=None, dependence=True) -> ArrivalService:
    svc = ArrivalService(enabled=False)

    def features(origin, dest, dep, airline):
        if airlines_seen is not None:
            airlines_seen.append(airline)
        return {"a": 0.1, "b": 0.2}

    svc.bundle = make_bundle()
    svc.provider = BookingLadderProvider(svc.bundle, features)
    svc.dependence = DependenceModel("R1", 0.3) if dependence else None
    svc.dependence_error = None if dependence else "FileNotFoundError: connection_dependence_v1.json not found"
    return svc
