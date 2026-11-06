"""Away / home / unknown for a flight, from its origin airport and the user's home airport (plan §4.2 rules).

No GPS and no location data. The only inputs are the flight's origin (the airport the user departs from) and the
home airport the user typed. The home airport is used only to compute the result and is never logged or stored.

Each `FlightRef` is one leg. For a round trip, the outbound leg starts at home and the return leg starts away.
For multi-city trips, each leg is evaluated on its own origin. Layovers and destinations do not matter here.
"""

from typing import Optional

from .schemas import Context

# Airports within one metro area count as "home" for each other (plan §4.2 and S-A5). Keep the codes in
# models/airports_ref.csv; tests check that.
METRO_GROUPS: dict[str, tuple[str, ...]] = {
    "NYC": ("JFK", "LGA", "EWR"),
    "CHI": ("ORD", "MDW"),
    "WAS": ("DCA", "IAD", "BWI"),
    "BAY": ("SFO", "OAK", "SJC"),
    "LA": ("LAX", "BUR", "LGB", "SNA", "ONT"),
    "MIA": ("MIA", "FLL"),
    "DFW": ("DFW", "DAL"),
    "IAH": ("IAH", "HOU"),
}

_METRO_OF: dict[str, str] = {code: group for group, codes in METRO_GROUPS.items() for code in codes}


def _norm(code: Optional[str]) -> Optional[str]:
    value = (code or "").strip().upper()
    return value or None


def same_metro(a: str, b: str) -> bool:
    """True when two IATA codes are the same airport or in the same metro group."""
    a, b = _norm(a), _norm(b)
    if a is None or b is None:
        return False
    if a == b:
        return True
    group_a = _METRO_OF.get(a)
    return group_a is not None and group_a == _METRO_OF.get(b)


def location(home_airport: Optional[str], origin: str) -> Context:
    """`home` when the origin is the home airport or in its metro group, `away` when it is not,
    `unknown` when no home airport is set (plan §4.2)."""
    origin_code = _norm(origin)
    if origin_code is None:
        raise ValueError("origin airport is required")
    home = _norm(home_airport)
    if home is None:
        return Context(location="unknown", reason="no_home_airport", home_airport=None)
    if same_metro(home, origin_code):
        return Context(location="home", reason="origin_is_home", home_airport=home)
    return Context(location="away", reason="origin_not_home", home_airport=home)
