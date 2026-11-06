"""Per-partner hand-off URL templates (plan section 3.6).

Pure functions: no environment access, no network. Affiliate decoration is
applied in ``copilot/links.py``. Nothing here has been checked by opening the
URL on a device yet; each format is annotated with its verification status.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlencode

AIRPORT_RE = re.compile(r"[A-Z]{3}")

@dataclass(frozen=True)
class Partner:
    id: str
    label: str
    # Env var suffix for AFFILIATE_<KEY>_PARAMS / _WRAP. None = no affiliate mechanism.
    affiliate_key: str | None
    # True = no usable plain link; skipped entirely unless affiliate config is set.
    affiliate_only: bool = False
    accepts_dropoff_address: bool = False


PARTNERS: dict[str, Partner] = {
    p.id: p
    for p in (
        Partner("google_flights", "Google Flights", None),
        Partner("kayak", "KAYAK", "KAYAK"),
        Partner("skyscanner", "Skyscanner", "SKYSCANNER"),
        Partner("expedia", "Expedia", "EXPEDIA"),
        Partner("travelpayouts", "Travelpayouts", "TRAVELPAYOUTS", affiliate_only=True),
        Partner("stay22", "Stay22", "STAY22", affiliate_only=True),
        Partner("booking", "Booking.com", "BOOKING"),
        Partner("google_maps", "Google Maps", None),
        Partner("uber", "Uber", "UBER", accepts_dropoff_address=True),
        Partner("lyft", "Lyft", "LYFT", accepts_dropoff_address=False),
    )
}

FLIGHT_PARTNERS = ("google_flights", "kayak", "skyscanner", "expedia", "travelpayouts")
HOTEL_PARTNERS = ("stay22", "booking", "expedia", "google_maps")
RIDE_PARTNERS = ("uber", "lyft")


def airport_code(value: str) -> str:
    code = (value or "").strip().upper()
    if not AIRPORT_RE.fullmatch(code):
        raise ValueError(f"not a 3-letter IATA airport code: {value!r}")
    return code


def check_latlon(lat: float, lon: float) -> None:
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("latitude and longitude must be finite")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("latitude/longitude out of range")


def check_stay(checkin: date, checkout: date) -> None:
    if checkout <= checkin:
        raise ValueError("checkout must be after checkin")


# --- flights -----------------------------------------------------------------

def google_flights_url(origin: str, destination: str, on: date) -> str:
    # Format from plan 3.6 (text query). UNVERIFIED by hand.
    q = f"Flights from {origin} to {destination} on {on.isoformat()}"
    return "https://www.google.com/travel/flights?" + urlencode({"q": q})


def kayak_flights_url(origin: str, destination: str, on: date) -> str:
    # Path format from plan 3.6. UNVERIFIED by hand.
    return f"https://www.kayak.com/flights/{origin}-{destination}/{on.isoformat()}"


def skyscanner_flights_url(origin: str, destination: str, on: date) -> str:
    # Path format from plan 3.6 (yyMMdd date, lowercase IATA). UNVERIFIED by hand.
    return (
        f"https://www.skyscanner.com/transport/flights/"
        f"{origin.lower()}/{destination.lower()}/{on:%y%m%d}/"
    )


def expedia_flights_url(origin: str, destination: str, on: date) -> str:
    # Flights-Search query format as remembered by the author; UNVERIFIED by hand.
    leg1 = f"from:{origin},to:{destination},departure:{on:%m/%d/%Y}TANYT"
    return "https://www.expedia.com/Flights-Search?" + urlencode(
        {"trip": "oneway", "leg1": leg1, "passengers": "adults:1", "mode": "search"},
        safe=":,/",
    )


def travelpayouts_flights_url(origin: str, destination: str, on: date) -> str:
    # Aviasales search path (DDMM date, passenger count suffix). Destination for the
    # Travelpayouts tracking wrapper. UNVERIFIED by hand.
    return f"https://www.aviasales.com/search/{origin}{on:%d%m}{destination}1"


# --- hotels ------------------------------------------------------------------

def stay22_hotels_url(lat: float, lon: float, checkin: date, checkout: date) -> str:
    # Map path and parameter names as remembered by the author; UNVERIFIED by hand.
    return "https://www.stay22.com/allez/map?" + urlencode(
        {"lat": lat, "lng": lon, "checkin": checkin.isoformat(),
         "checkout": checkout.isoformat(), "adults": 1}
    )


def booking_hotels_url(lat: float, lon: float, checkin: date, checkout: date) -> str:
    # Format from plan 3.6. UNVERIFIED by hand.
    return "https://www.booking.com/searchresults.html?" + urlencode(
        {"latitude": lat, "longitude": lon, "checkin": checkin.isoformat(),
         "checkout": checkout.isoformat(), "group_adults": 1}
    )


def expedia_hotels_url(lat: float, lon: float, checkin: date, checkout: date) -> str:
    # Hotel-Search latLong format as remembered by the author; UNVERIFIED by hand.
    return "https://www.expedia.com/Hotel-Search?" + urlencode(
        {"latLong": f"{lat},{lon}", "startDate": f"{checkin:%m/%d/%Y}",
         "endDate": f"{checkout:%m/%d/%Y}", "adults": 1, "rooms": 1},
        safe=",/",
    )


def google_maps_hotels_url(airport: str) -> str:
    # Plan 3.6 example ("hotels near JFK airport"). UNVERIFIED by hand.
    return "https://www.google.com/maps/search/hotels+near+" + airport + "+airport"


# --- rides -------------------------------------------------------------------

def uber_url(dropoff: tuple[float, float] | None = None) -> str:
    # Documented Uber universal link (plan 3.6). Dropoff coordinates optional.
    params: dict[str, object] = {"action": "setPickup", "pickup": "my_location"}
    if dropoff is not None:
        params["dropoff[latitude]"], params["dropoff[longitude]"] = dropoff
    return "https://m.uber.com/ul/?" + urlencode(params, safe="[]")


def lyft_url(dropoff: tuple[float, float] | None = None) -> str:
    # Plan 3.6 format. Dropoff parameter names as remembered by the author; UNVERIFIED on a phone.
    params: dict[str, object] = {"id": "lyft"}
    if dropoff is not None:
        params["destination[latitude]"], params["destination[longitude]"] = dropoff
    return "https://lyft.com/ride?" + urlencode(params, safe="[]")
