"""Pairing priors built from store days strictly before a measured day (protocol: linkage/PROTOCOL_CHAIN.md).

A pairing is two consecutive legs of the same aircraft (by truth) at one airport: same carrier, A arrives where B
departs, B no more than 18 h after A. The prior for a flight number is "what followed it (next) / preceded it (prev)
on the same weekday a week earlier, else the day before". Only operating days before `before_day` are used, so the
measured day never sees its own pairings.

Operating day = the UTC date of (scheduled out - 6 h), so one evening's flights stay on one day.
"""
from collections import defaultdict
from typing import Optional

MAX_GAP_S = 18 * 3600


def op_day(so: float) -> int:
    """Epoch-day number of the operating day of a scheduled-out time."""
    return int((so - 6 * 3600) // 86400)


class Priors:
    def __init__(self, nxt: dict, prv: dict):
        self._nxt, self._prv = nxt, prv

    @staticmethod
    def _look(tbl, carrier, fn, airport, day) -> Optional[str]:
        v = tbl.get((carrier, fn, airport, day - 7))
        return v if v is not None else tbl.get((carrier, fn, airport, day - 1))

    def next_fn(self, carrier, fn, dest, day) -> Optional[str]:
        """Flight number that usually departs `dest` after `fn` arrives there."""
        return self._look(self._nxt, carrier, fn, dest, day)

    def prev_fn(self, carrier, fn, origin, day) -> Optional[str]:
        """Flight number that usually arrives at `origin` before `fn` departs."""
        return self._look(self._prv, carrier, fn, origin, day)


def build(legs, truth: dict, before_day: int) -> Priors:
    """legs: iterable of chain_rules.Leg. truth: leg id -> aircraft key. Uses legs whose operating day is
    strictly before `before_day`."""
    by_ac = defaultdict(list)
    for lg in legs:
        k = truth.get(lg.id)
        if k is not None and op_day(lg.so) < before_day:
            by_ac[k].append(lg)
    nxt, prv = {}, {}
    for seq in by_ac.values():
        seq.sort(key=lambda g: g.so)
        for a, b in zip(seq, seq[1:]):
            if a.carrier == b.carrier and a.dest == b.origin and 0 < b.so - a.so <= MAX_GAP_S and a.si <= b.so + 300:
                nxt[(a.carrier, a.fn, a.dest, op_day(a.so))] = b.fn
                prv[(b.carrier, b.fn, b.origin, op_day(b.so))] = a.fn
    return Priors(nxt, prv)
