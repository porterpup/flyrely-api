"""Leg forecasters: anything that turns legs into `LegForecast`s. The connection engine does not care where the
marginals come from, so run-up or live-ETA providers can replace the booking ladder later (plan section 2.2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Protocol

import numpy as np

from .connections import LegForecast
from .ladder import ARR_THRESHOLDS, DEP_THRESHOLDS, Ladder, LadderBundle


@dataclass(frozen=True)
class LegSpec:
    origin: str
    destination: str
    departure_time: datetime            # local, naive
    airline: Optional[str] = None       # operating carrier (the API resolves codeshares before this point)


class LegForecaster(Protocol):
    def forecast(self, legs: list[LegSpec], as_of: Optional[datetime] = None) -> list[LegForecast]: ...


class BookingLadderProvider:
    """Booking-time ladders: the 18 schedule/route features only (provider "booking-ladder-v1")."""

    provider = "booking-ladder-v1"
    horizon = "booking"

    def __init__(self, bundle: LadderBundle, feature_fn: Optional[Callable] = None):
        self.bundle = bundle
        self._feature_fn = feature_fn

    def _features(self, leg: LegSpec) -> dict:
        fn = self._feature_fn
        if fn is None:
            from main import build_features      # serving feature builder (verified against training in the parity tests)
            fn = build_features
        return fn(leg.origin, leg.destination, leg.departure_time, leg.airline)

    def forecast(self, legs: list[LegSpec], as_of: Optional[datetime] = None) -> list[LegForecast]:
        X = np.array([[self._features(l)[k] for k in self.bundle.feature_names] for l in legs], dtype=np.float32)
        r = self.bundle.predict_arrays(X)
        return [LegForecast(Ladder(tuple(ARR_THRESHOLDS), tuple(r["arr"][i]), self.bundle.arr_tail),
                            Ladder(tuple(DEP_THRESHOLDS), tuple(r["dep"][i]), self.bundle.dep_tail),
                            float(r["cancel"][i]), self.provider, self.horizon)
                for i in range(len(legs))]
