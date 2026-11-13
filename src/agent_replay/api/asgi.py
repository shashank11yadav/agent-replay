"""ASGI entry point: ``uvicorn agent_replay.api.asgi:app``."""

from __future__ import annotations

from agent_replay.api.app import create_app
from agent_replay.config import Settings

app = create_app(Settings.model_validate({}))
