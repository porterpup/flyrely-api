"""Pydantic models for every /copilot contract in the plan §4. The app mirrors these in src/copilot/types.ts."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high"]
DisruptionState = Literal["at_risk", "cancelled", "rebooked"]
FlightStatus = Literal["unknown", "scheduled", "delayed", "cancelled", "departed", "arrived"]


class FlightRef(BaseModel):
    airline: Optional[str] = Field(default=None, min_length=2, max_length=3, pattern=r"^[A-Za-z0-9]{2,3}$")
    flight_number: str = Field(min_length=1, max_length=12, pattern=r"^\s*(?:[A-Za-z0-9]{2}|[A-Za-z]{3})\s?\d{1,4}[A-Za-z]?\s*$|^\s*\d{1,4}[A-Za-z]?\s*$")
    origin: str = Field(min_length=3, max_length=3, pattern=r"^[A-Za-z]{3}$")
    destination: str = Field(min_length=3, max_length=3, pattern=r"^[A-Za-z]{3}$")
    departure_time: str = Field(max_length=40)  # local wall-clock ISO, as in /predict


# 4.1 /copilot/config
class PushConfig(BaseModel):
    enabled: bool
    vapid_public_key: Optional[str] = None


class ConfigResponse(BaseModel):
    enabled: bool
    push: PushConfig
    status_source: str
    affiliate_partners_configured: list[str]
    schedule_as_of: Optional[str] = None


# 4.2 /copilot/options
class OptionsRequest(BaseModel):
    flight: FlightRef
    state: DisruptionState
    rebooked: Optional[Literal["today", "later"]] = None
    home_airport: Optional[str] = Field(default=None, max_length=8)
    max_alternatives: int = Field(default=8, ge=1, le=15)


class Handoff(BaseModel):
    partner: str
    label: str
    url: str
    is_affiliate: bool


class RideLink(Handoff):
    accepts_dropoff_address: bool


class Context(BaseModel):
    location: Literal["home", "away", "unknown"]
    reason: Literal["origin_is_home", "origin_not_home", "no_home_airport"]
    home_airport: Optional[str] = None


class AirlineInfo(BaseModel):
    code: str
    name: str
    manage_booking_url: str
    phone: str
    phone_display: str
    verified_on: Optional[str] = None
    verification: Literal["verified", "official_source_unconfirmed"] = "verified"
    other_channels: dict[str, str] = Field(default_factory=dict)


class AirlineNotice(BaseModel):
    """Shown instead of contact details when `airline` is null but the carrier has a note or status page."""
    code: str
    name: str
    note: Optional[str] = None
    status_url: Optional[str] = None


class RefundRights(BaseModel):
    applies: bool
    jurisdiction: Optional[str] = None
    content_key: Optional[str] = None
    # Additive (v2). international: either end is outside the 50 states + DC (airports_ref has no state for
    # territories and non-US airports). eu_possible: None = not determinable (airports_ref is US-only).
    international: bool = False
    eu_possible: Optional[bool] = None


class AlternativeItem(BaseModel):
    marketing_airline: str
    flight_number: str
    operating_airline: Optional[str] = None
    operating_airline_name: Optional[str] = None
    origin: str
    destination: str
    dep_local: str
    arr_local: Optional[str] = None
    delay_probability: Optional[float] = None
    risk_level: Optional[RiskLevel] = None
    days_seen: int
    handoffs: list[Handoff]


class Alternatives(BaseModel):
    source: str
    schedule_as_of: Optional[str] = None
    date: str
    items: list[AlternativeItem]
    search_handoffs: list[Handoff]
    warnings: list[str] = Field(default_factory=list)


class Hotel(BaseModel):
    airport: str
    check_in: str
    check_out: str
    links: list[Handoff]


class Ride(BaseModel):
    purpose: Optional[Literal["to_hotel", "home"]] = None
    links: list[RideLink]


class Disclosure(BaseModel):
    required: bool


class OptionsResponse(BaseModel):
    state: DisruptionState
    context: Context
    airline: Optional[AirlineInfo] = None
    airline_notice: Optional[AirlineNotice] = None
    refund_rights: RefundRights
    alternatives: Alternatives
    hotel: Optional[Hotel] = None
    ride: Ride
    disclosure: Disclosure
    warnings: list[str] = Field(default_factory=list)


# 4.3 /copilot/status
class StatusRequest(BaseModel):
    flight: FlightRef


class StatusResponse(BaseModel):
    status: FlightStatus
    source: str
    as_of: Optional[str] = None
    detail: Optional[str] = None


# 4.6 push payload
class PushPayload(BaseModel):
    v: Literal[1] = 1
    kind: Literal["at_risk_heads_up", "cancelled", "test"]
    title: str
    body: str
    url: str
    client_flight_id: str


# 4.4 / 4.5 push subscribe / unsubscribe
class PushKeys(BaseModel):
    p256dh: str = Field(min_length=1, max_length=256)
    auth: str = Field(min_length=1, max_length=128)


class PushSubscription(BaseModel):
    endpoint: str = Field(min_length=1, max_length=2048)
    keys: PushKeys


class SubscribeRequest(BaseModel):
    subscription: PushSubscription
    flight: FlightRef
    client_flight_id: Optional[str] = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


class SubscribeResponse(BaseModel):
    watch_id: str
    expires_at: str


class UnsubscribeRequest(BaseModel):
    endpoint: str = Field(min_length=1, max_length=2048)
    watch_id: Optional[str] = Field(default=None, max_length=64)


class UnsubscribeResponse(BaseModel):
    removed: int
