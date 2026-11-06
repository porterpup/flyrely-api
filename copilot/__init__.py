"""Disruption copilot (docs/PLAN_DISRUPTION_COPILOT.md). Mounted only behind FLYRELY_FEATURE_DISRUPTION_COPILOT=1.

This package must never import `main`: the scorer is injected through `mount(app, score=...)`.
"""
from datetime import datetime
from typing import Callable, Optional

from fastapi import FastAPI

# score(origin, destination, dep_time, airline) -> (delay_probability, risk_level); raises on failure.
Scorer = Callable[[str, str, datetime, Optional[str]], tuple[float, str]]


def mount(app: FastAPI, score: Optional[Scorer] = None) -> None:
    """Attach the /copilot routes to `app` and keep the scorer for later steps (S-A6)."""
    from .router import build_router

    app.state.copilot_score = score
    app.include_router(build_router())
    from . import watcher   # starts nothing unless all three flags are on, and never under pytest
    watcher.attach(app, score)


__all__ = ["mount", "Scorer"]
