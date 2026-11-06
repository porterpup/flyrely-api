"""Copilot feature flags and config (plan §5). Read from the environment at call time."""
import os
from dataclasses import dataclass
from typing import Optional

FLAG_MASTER = "FLYRELY_FEATURE_DISRUPTION_COPILOT"
FLAG_PUSH = "FLYRELY_COPILOT_PUSH"
STATUS_SOURCES = ("none", "swim")
# Partners that can carry affiliate config (plan §3.6).
AFFILIATE_PARTNERS = ("KAYAK", "SKYSCANNER", "EXPEDIA", "TRAVELPAYOUTS", "STAY22", "BOOKING", "UBER", "LYFT")


def feature_enabled() -> bool:
    return os.getenv(FLAG_MASTER) == "1"


def push_enabled() -> bool:
    return feature_enabled() and os.getenv(FLAG_PUSH) == "1"


def status_source() -> str:
    value = (os.getenv("FLYRELY_COPILOT_STATUS_SOURCE") or "none").strip().lower()
    return value if value in STATUS_SOURCES else "none"


def vapid_public_key() -> Optional[str]:
    return os.getenv("FLYRELY_COPILOT_VAPID_PUBLIC_KEY") or None


def affiliate_partners_configured() -> list[str]:
    """Lower-case partner names with an AFFILIATE_<P>_PARAMS or _WRAP value set (values never exposed)."""
    return [
        p.lower() for p in AFFILIATE_PARTNERS
        if os.getenv(f"AFFILIATE_{p}_PARAMS") or os.getenv(f"AFFILIATE_{p}_WRAP")
    ]


def schedule_as_of() -> Optional[str]:
    """Latest month of the typical-schedule artifact (YYYY-MM), or None if the artifact is missing."""
    from .options import default_schedule  # lazy: reading the artifact must not happen at import
    return default_schedule().schedule_as_of()


@dataclass(frozen=True)
class CopilotConfig:
    enabled: bool
    push_enabled: bool
    vapid_public_key: Optional[str]
    status_source: str
    affiliate_partners_configured: list[str]
    schedule_as_of: Optional[str]


def load() -> CopilotConfig:
    push = push_enabled()
    return CopilotConfig(
        enabled=feature_enabled(),
        push_enabled=push,
        vapid_public_key=vapid_public_key() if push else None,
        status_source=status_source(),
        affiliate_partners_configured=affiliate_partners_configured(),
        schedule_as_of=schedule_as_of(),
    )
