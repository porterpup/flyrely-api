"""Pure rules for the "anchor walk" forward chaining (protocol: linkage/PROTOCOL_CHAIN.md). No I/O, stdlib only.

An aircraft is identified (by an as-of fact) on some leg of its day. The walk moves forward from that leg, one
departure per step, choosing at each step, in order:
  1. a departure already filed with the same aircraft                        ("direct", high confidence)
  2. else the same flight number continuing, carrier on the M2 allow-list    ("m2", high)
  3. else the pairing prior (the flight number that usually follows)         ("prior", high only when the
     previous step matched the prior line)
A candidate must be the same carrier, leave the airport the aircraft is at, be the same aircraft-type family, and
must not already be filed with another aircraft. At most MAX_STEPS steps, and no step scheduled more than
MAX_SPAN_S after the anchor's scheduled departure.

Times are epoch seconds (float). Aircraft keys are opaque strings and are never logged here.
"""
from bisect import bisect_left
from typing import Callable, NamedTuple, Optional, Sequence

MAX_STEPS = 6
MAX_SPAN_S = 18 * 3600

# M2 (same flight number continuing) allow-list, ICAO carrier codes: WN DL AA OO MQ 9E OH PT YX AS QX.
# UA, G4, G7, C5, YV are excluded (BTS calibration T4, backtest/REPORT_L1_BTS_CALIB.md); others have too few candidates.
M2_ALLOW = frozenset({"SWA", "DAL", "AAL", "SKW", "ENY", "EDV", "JIA", "PDT", "RPA", "ASA", "QXE"})

_FAMILIES = {
    "B737": "B731 B732 B733 B734 B735 B736 B737 B738 B739 B37M B38M B39M B3XM",
    "A320": "A318 A319 A320 A321 A19N A20N A21N",
    "A220": "BCS1 BCS3 A220",
    "E170": "E170 E75L E75S E175 E17L",
    "E190": "E190 E195 E290 E295 E19L",
    "E145": "E145 E135 E45X E140",
    "CRJ2": "CRJ1 CRJ2",
    "CRJ7": "CRJ7 CRJ9 CRJX",
    "B717": "B712",
    "B757": "B752 B753",
    "B767": "B762 B763 B764",
    "B777": "B772 B77L B773 B77W",
    "B787": "B788 B789 B78X",
    "A330": "A332 A333 A338 A339",
    "A350": "A359 A35K",
    "MD80": "MD80 MD81 MD82 MD83 MD87 MD88 MD90",
    "DH8": "DH8A DH8B DH8C DH8D",
}
FAMILY_OF = {code: fam for fam, codes in _FAMILIES.items() for code in codes.split()}


def type_family(code) -> str:
    """ICAO aircraft type -> family ('' when unknown; unknown matches anything)."""
    c = (code or "").strip().upper()
    return FAMILY_OF.get(c, c)


def family_ok(a: str, b: str) -> bool:
    return not a or not b or a == b


class Leg(NamedTuple):
    id: str
    carrier: str     # operating carrier, ICAO
    fn: str          # flight number
    origin: str
    dest: str
    so: float        # scheduled out
    si: float        # scheduled in
    fam: str         # type_family(aircraft_type)


class Step(NamedTuple):
    leg: Leg
    method: str      # 'direct' | 'm2' | 'prior'
    high: bool
    matched_prior: bool


Ident = Callable[[str], Optional[str]]   # leg id -> aircraft key known at the cutoff, or None


def excluded(cand: Leg, aircraft: str, ident: Ident) -> bool:
    """A departure already filed with a different aircraft cannot be this aircraft's next flight."""
    k = ident(cand.id)
    return k is not None and k != aircraft


def next_step(cur: Leg, aircraft: str, fam: str, cands: Sequence[Leg], ident: Ident,
              prior_fn: Optional[str], prev_matched: bool) -> Optional[Step]:
    """One forward step from `cur`. `cands` are the carrier's departures from cur.dest, sorted by `so`, that
    leave no earlier than cur's scheduled arrival. Returns None when no rule applies."""
    for c in cands:
        if ident(c.id) == aircraft and family_ok(fam, c.fam):
            return Step(c, "direct", True, prior_fn is not None and c.fn == prior_fn)
    if cur.carrier in M2_ALLOW:
        for c in cands:
            if c.fn == cur.fn and not excluded(c, aircraft, ident) and family_ok(fam, c.fam):
                return Step(c, "m2", True, prior_fn is not None and c.fn == prior_fn)
    if prior_fn is not None:
        for c in cands:
            if c.fn == prior_fn and not excluded(c, aircraft, ident) and family_ok(fam, c.fam):
                return Step(c, "prior", prev_matched, True)
    return None


def walk(anchor: Leg, aircraft: str, departures: Callable[[str, str], Sequence[Leg]], ident: Ident,
         prior_next: Callable[[Leg], Optional[str]], prev_matched: bool = False,
         max_steps: int = MAX_STEPS, max_span_s: float = MAX_SPAN_S) -> list:
    """Walk forward from `anchor`. `departures(carrier, airport)` returns that carrier's legs from that airport
    sorted by `so`. Returns the Step list (possibly empty). `prev_matched`: whether the anchor itself followed
    the prior line (decides if a first prior step is high confidence)."""
    out, cur, fam = [], anchor, anchor.fam
    for _ in range(max_steps):
        deps = departures(cur.carrier, cur.dest)
        lo = bisect_left(deps, cur.si, key=lambda d: d.so)
        cands = []
        for d in deps[lo:]:
            if d.so - anchor.so > max_span_s:
                break
            if d.id != cur.id:
                cands.append(d)
        st = next_step(cur, aircraft, fam, cands, ident, prior_next(cur), prev_matched)
        if st is None:
            break
        out.append(st)
        prev_matched = st.matched_prior
        cur = st.leg
        fam = cur.fam or fam
    return out


class AsOfState:
    """Incremental as-of view of aircraft identity. `obs` is a list of (recv_epoch, leg_id, key, source).
    advance(cutoff) applies every fact with recv <= cutoff (a fact 1 s later is ignored); cutoffs must not go back.
    ident(leg) is the latest key by recv, any source; ident_sfdps(leg) is the latest SFDPS key (M1)."""

    def __init__(self, obs):
        self.obs = sorted(obs, key=lambda o: o[0])
        self.i = 0
        self.any: dict = {}
        self.sfdps: dict = {}
        self.legs_of: dict = {}

    def advance(self, cutoff: float):
        obs, n = self.obs, len(self.obs)
        while self.i < n and obs[self.i][0] <= cutoff:
            _, leg, key, src = obs[self.i]
            self.i += 1
            old = self.any.get(leg)
            if old != key:
                if old is not None:
                    self.legs_of[old].discard(leg)
                self.any[leg] = key
                self.legs_of.setdefault(key, set()).add(leg)
            if src == "sfdps":
                self.sfdps[leg] = key
        return self

    def ident(self, leg: str):
        return self.any.get(leg)

    def ident_sfdps(self, leg: str):
        return self.sfdps.get(leg)
