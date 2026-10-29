"""
Ticketed (marketing) flight -> the carrier that actually operates it.

FlyRely models delays at the operating-carrier level (SkyWest, Envoy, ...), but
customers know the airline that sold the ticket (Delta, American, ...).
The table is built by scripts/build_codeshare_map.py from BTS Marketing Carrier data.
"""

import csv
import re
from pathlib import Path
from typing import Optional

MAP_PATH = Path(__file__).resolve().parent / "models" / "codeshare_map.csv"

OPERATOR_NAMES = {
    "9E": "Endeavor Air", "AA": "American Airlines", "AS": "Alaska Airlines", "B6": "JetBlue",
    "C5": "CommutAir", "DL": "Delta Air Lines", "F9": "Frontier", "G4": "Allegiant", "G7": "GoJet",
    "HA": "Hawaiian Airlines", "MQ": "Envoy Air", "NK": "Spirit", "OH": "PSA Airlines",
    "OO": "SkyWest Airlines", "PT": "Piedmont Airlines", "QX": "Horizon Air", "SY": "Sun Country",
    "UA": "United Airlines", "WN": "Southwest", "YV": "Mesa Airlines", "YX": "Republic Airways",
    "ZW": "Air Wisconsin",
}

_by_route: dict[tuple, dict] = {}
_by_number: dict[tuple, dict] = {}


def _load():
    if _by_route or not MAP_PATH.exists():
        return
    with MAP_PATH.open() as f:
        for r in csv.DictReader(f):
            entry = {"operating_airline": r["operating_airline"],
                     "operating_flight_number": r["operating_flight_number"],
                     "share": float(r["share"]), "n": int(r["n"])}
            _by_route[(r["marketing_airline"], r["marketing_flight_number"], r["origin"], r["dest"])] = entry
            key = (r["marketing_airline"], r["marketing_flight_number"])
            if key not in _by_number or entry["n"] > _by_number[key]["n"]:
                _by_number[key] = entry


def parse_flight(airline: Optional[str], flight_number: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Accept 'DL1234', 'DL 1234' or airline='DL' + '1234'; return ('DL', '1234')."""
    if flight_number:
        # Airline codes have at least one letter (DL, B6, 9E), so "4632" is all flight number
        m = re.fullmatch(r"\s*([A-Z]{2}|[A-Z]\d|\d[A-Z])?\s*0*(\d{1,4})\s*", flight_number.upper())
        if m:
            return (m.group(1) or airline), m.group(2)
    return airline, None


def resolve(airline: Optional[str], flight_number: Optional[str],
            origin: Optional[str] = None, dest: Optional[str] = None) -> Optional[dict]:
    """Operating carrier for a ticketed flight, or None if unknown."""
    _load()
    airline, number = parse_flight(airline, flight_number)
    if not airline or not number:
        return None
    hit = _by_route.get((airline, number, origin, dest)) or _by_number.get((airline, number))
    if not hit:
        return None
    return {**hit, "marketing_airline": airline, "marketing_flight_number": number,
            "operating_airline_name": OPERATOR_NAMES.get(hit["operating_airline"], hit["operating_airline"])}
