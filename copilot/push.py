"""Web push sender (plan §6): pywebpush + VAPID, imported lazily so the flag-off app never loads it.

A sender is any callable `send(subscription: dict, payload: dict) -> None` that raises `PushGone` when the push
service says the subscription no longer exists (404/410) and `PushError` for anything else. Nothing here logs the
endpoint, keys or payload.
"""
import json
import os
from typing import Callable, Optional

from . import push_store

TTL_S = 6 * 3600   # a heads-up older than this is no longer useful


class PushError(Exception):
    """The send failed; retry later. The message never contains the endpoint."""


class PushGone(PushError):
    """404/410 from the push service: delete the subscription."""


Sender = Callable[[dict, dict], None]


def build_payload(kind: str, *, title: str, body: str, client_flight_id: Optional[str]) -> dict:
    """Plan §4.6. Never includes the home airport or a partner link."""
    return {"v": 1, "kind": kind, "title": title, "body": body,
            "url": f"/flights/{client_flight_id}" if client_flight_id else "/",
            "client_flight_id": client_flight_id}


def payload_for(watch: push_store.Watch, kind: str, probability: Optional[float] = None) -> dict:
    flight = watch.flight_number if watch.flight_number[:2].isalpha() else f"{watch.airline}{watch.flight_number}"
    if kind == push_store.KIND_CANCELLED:
        return build_payload(kind, title=f"{flight} was cancelled",
                             body="Tap for your options to rebook.", client_flight_id=watch.client_flight_id)
    pct = f"{round((probability or 0) * 100)}% chance of a 15+ min delay. " if probability is not None else ""
    return build_payload(kind, title=f"{flight} departs within 24 hours: high delay risk",
                         body=f"{pct}Tap for your Plan B.", client_flight_id=watch.client_flight_id)


def vapid_private_key() -> Optional[str]:
    return os.getenv("FLYRELY_COPILOT_VAPID_PRIVATE_KEY") or None


def vapid_subject() -> Optional[str]:
    return os.getenv("FLYRELY_COPILOT_VAPID_SUBJECT") or None


def vapid_configured() -> bool:
    return bool(vapid_private_key() and vapid_subject() and os.getenv("FLYRELY_COPILOT_VAPID_PUBLIC_KEY"))


def webpush_sender(private_key: Optional[str] = None, subject: Optional[str] = None, timeout: float = 10.0) -> Sender:
    """The real sender. Keys come from the environment at call time unless given."""
    def send(subscription: dict, payload: dict) -> None:
        key, sub = private_key or vapid_private_key(), subject or vapid_subject()
        if not key or not sub:
            raise PushError("VAPID keys are not configured")
        from pywebpush import WebPushException, webpush   # lazy
        try:
            webpush(subscription_info=subscription, data=json.dumps(payload), vapid_private_key=key,
                    vapid_claims={"sub": sub}, ttl=TTL_S, timeout=timeout)
        except WebPushException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (404, 410):
                raise PushGone(f"push service returned {status}") from None
            raise PushError(f"push service error{f' {status}' if status else ''}") from None
        except Exception as e:   # network, bad key, ...: type only, the text may contain the URL
            raise PushError(type(e).__name__) from None
    return send
