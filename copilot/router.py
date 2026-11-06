"""/copilot endpoints. S-A1 ships GET /copilot/config only; /options and /status arrive in S-A6."""
from fastapi import APIRouter

from . import config
from .schemas import ConfigResponse, PushConfig


def build_router() -> APIRouter:
    router = APIRouter(prefix="/copilot", tags=["copilot"])

    @router.get("/config", response_model=ConfigResponse)
    async def get_config() -> ConfigResponse:
        cfg = config.load()
        return ConfigResponse(
            enabled=cfg.enabled,
            push=PushConfig(enabled=cfg.push_enabled, vapid_public_key=cfg.vapid_public_key),
            status_source=cfg.status_source,
            affiliate_partners_configured=cfg.affiliate_partners_configured,
            schedule_as_of=cfg.schedule_as_of,
        )

    return router
