"""Flight status sources for POST /copilot/status (plan §3.3, §4.3).

- `NullStatusSource` (FLYRELY_COPILOT_STATUS_SOURCE unset or "none") always reports `unknown`.
- `SwimLiveStateSource` (FLYRELY_COPILOT_STATUS_SOURCE=swim) is a stub. It raises `SourceNotConfigured` until
  roadmap R2/R3 (always-on archiver and live-state extractor into Postgres) exist. Callers must catch it.

The copilot never reads a status it cannot back with data. Nothing here calls the network.
"""

from typing import Optional, Protocol

from . import config
from .schemas import FlightRef, StatusResponse


class SourceNotConfigured(RuntimeError):
    """The status source is selected but has no backing data yet (SWIM live state, roadmap R3)."""


class StatusSource(Protocol):
    source: str

    def status(self, flight: FlightRef) -> StatusResponse: ...


class NullStatusSource:
    source = "none"

    def status(self, flight: FlightRef) -> StatusResponse:
        return StatusResponse(status="unknown", source=self.source, as_of=None, detail=None)


class SwimLiveStateSource:
    """Stub until roadmap R3 (live-state extractor into Postgres, fed by the R2 archiver).

    When R3 exists, implement `status()` as a query on its table (plan §3.3). Until then it raises.
    """

    source = "swim"

    def status(self, flight: FlightRef) -> StatusResponse:
        raise SourceNotConfigured(
            "SWIM live-state status is not available until roadmap R3 (live-state extractor) exists"
        )


def get_status_source(name: Optional[str] = None) -> StatusSource:
    """Factory. `name` defaults to FLYRELY_COPILOT_STATUS_SOURCE (read at call time)."""
    value = (config.status_source() if name is None else name).strip().lower()
    if value == "none":
        return NullStatusSource()
    if value == "swim":
        return SwimLiveStateSource()
    raise ValueError(f"unknown copilot status source: {value!r}")
