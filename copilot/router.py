"""/copilot endpoints (plan §4): GET /config, POST /options, POST /status."""
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from . import config, options, push_store, status as status_mod
from .schemas import (ConfigResponse, OptionsRequest, OptionsResponse, PushConfig, StatusRequest, StatusResponse,
                      SubscribeRequest, SubscribeResponse, UnsubscribeRequest, UnsubscribeResponse)


def _push_store(request: Request) -> push_store.PushStore:
    """404 unless FLYRELY_COPILOT_PUSH=1 (checked per request, like the other flags); lazily one shared store."""
    if not config.push_enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    state = request.app.state
    if getattr(state, "copilot_push_store", None) is None:
        state.copilot_push_store = push_store.PushStore()
    return state.copilot_push_store


def build_router() -> APIRouter:
    router = APIRouter(prefix="/copilot", tags=["copilot"])

    @router.get("/config", response_model=ConfigResponse)
    def get_config() -> ConfigResponse:
        cfg = config.load()
        return ConfigResponse(
            enabled=cfg.enabled,
            push=PushConfig(enabled=cfg.push_enabled, vapid_public_key=cfg.vapid_public_key),
            status_source=cfg.status_source,
            affiliate_partners_configured=cfg.affiliate_partners_configured,
            schedule_as_of=cfg.schedule_as_of,
        )

    @router.post("/options", response_model=OptionsResponse)
    def post_options(body: OptionsRequest, request: Request) -> OptionsResponse:
        state = request.app.state
        kwargs = {}
        if getattr(state, "copilot_now", None) is not None:   # test hook: frozen clock
            kwargs["now"] = state.copilot_now
        return options.compose(
            body,
            score=getattr(state, "copilot_score", None),
            schedule=getattr(state, "copilot_schedule", None) or options.default_schedule(),
            **kwargs,
        )

    @router.post("/status", response_model=StatusResponse)
    def post_status(body: StatusRequest, request: Request) -> StatusResponse:
        source = getattr(request.app.state, "copilot_status_source", None) or status_mod.get_status_source()
        try:
            return source.status(body.flight)
        except status_mod.SourceNotConfigured:
            return StatusResponse(status="unknown", source=source.source, as_of=None, detail="source_not_configured")

    @router.post("/push/subscribe", response_model=SubscribeResponse, status_code=201)
    def post_push_subscribe(body: SubscribeRequest, request: Request) -> SubscribeResponse:
        store = _push_store(request)
        sub, flight = body.subscription, body.flight
        if not push_store.endpoint_allowed(sub.endpoint):
            raise HTTPException(status_code=400, detail="unsupported push endpoint")
        try:
            departure = push_store.departure_dt(flight.departure_time)
        except ValueError:
            raise HTTPException(status_code=422, detail="invalid departure_time")
        now = getattr(request.app.state, "copilot_now", None)
        now = now() if now is not None else datetime.now(timezone.utc)
        if push_store.expiry_for(departure) <= now:
            raise HTTPException(status_code=422, detail="flight is in the past")
        airline = (flight.airline or "").strip().upper()
        number = flight.flight_number.strip().upper()
        if not number or len(number) > 12:
            raise HTTPException(status_code=422, detail="invalid flight_number")
        try:
            watch = store.subscribe(
                endpoint=sub.endpoint, p256dh=sub.keys.p256dh, auth=sub.keys.auth, airline=airline,
                flight_number=number, origin=flight.origin.strip().upper(),
                destination=flight.destination.strip().upper(), departure=departure,
                client_flight_id=body.client_flight_id, now=now)
        except push_store.TooManyWatches:
            raise HTTPException(status_code=429, detail="too many active alerts for this device")
        return SubscribeResponse(watch_id=watch.watch_id,
                                 expires_at=watch.expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"))

    @router.post("/push/unsubscribe", response_model=UnsubscribeResponse)
    def post_push_unsubscribe(body: UnsubscribeRequest, request: Request) -> UnsubscribeResponse:
        store = _push_store(request)
        return UnsubscribeResponse(removed=store.unsubscribe(body.endpoint, body.watch_id))

    return router
