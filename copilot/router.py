"""/copilot endpoints (plan §4): GET /config, POST /options, POST /status."""
from fastapi import APIRouter, Request

from . import config, options, status as status_mod
from .schemas import ConfigResponse, OptionsRequest, OptionsResponse, PushConfig, StatusRequest, StatusResponse


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

    return router
