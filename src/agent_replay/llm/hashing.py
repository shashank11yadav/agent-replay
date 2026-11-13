"""Deterministic request hashing and tool-call id helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from agent_replay.llm.types import Message, ModelRequest

CACHE_SCHEMA_VERSION = 1


def _strip_signatures(dumped: dict[str, Any]) -> dict[str, Any]:
    """Drop opaque tool-call signatures so cache keys depend only on content."""
    for message in dumped.get("messages", []):
        for call in message.get("tool_calls", []):
            call.pop("signature", None)
    return dumped


def request_hash(request: ModelRequest) -> str:
    """Return the SHA-256 hex digest of the canonical request plus the cache schema version."""
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "request": _strip_signatures(request.model_dump(mode="json")),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def next_step(messages: Sequence[Message]) -> int:
    """The step number of the model turn being generated (1-based)."""
    return 1 + sum(1 for m in messages if m.role == "model")


def make_tool_call_id(step: int, index: int) -> str:
    return f"call_{step}_{index}"
