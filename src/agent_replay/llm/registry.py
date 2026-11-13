"""Provider registry: each model provider registers a ``ProviderSpec``.

The spec is plain data plus a ``build`` callable, so ``config.py`` can resolve per-provider
defaults without importing any provider SDK (built-in builders import theirs lazily).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_replay.llm.provider import ModelProvider

if TYPE_CHECKING:
    from agent_replay.config import Settings

SCRIPT_PATH_ENV = "AGENT_REPLAY_SCRIPT_PATH"

# The committed baseline and regression fixtures were recorded with this provider, so the
# default replays them from the cache.
DEFAULT_PROVIDER = "claude_code"

Prices = tuple[float, float]


@dataclass(frozen=True)
class ProviderSpec:
    """How to build one provider and what its defaults are."""

    name: str
    # Returns the live provider, or None in replay mode when it has no credential or tool.
    # Raises ValueError in record mode when it cannot record.
    build: Callable[[Settings, str], ModelProvider | None]
    needs_credential_to_record: bool
    default_model: str
    default_judge_model: str | None  # None = same as the effective agent model
    default_prices: Prices  # (input, output) USD per 1M tokens
    model_prices: Mapping[str, Prices] = field(default_factory=dict)
    rate_limited: bool = False


_specs: dict[str, ProviderSpec] = {}


def register(spec: ProviderSpec) -> None:
    if spec.name in _specs:
        raise ValueError(f"Provider already registered: {spec.name!r}")
    _specs[spec.name] = spec


def unregister(name: str) -> None:
    if name not in _specs:
        raise ValueError(f"Provider not registered: {name!r}")
    del _specs[name]


def get_provider_spec(name: str) -> ProviderSpec:
    try:
        return _specs[name]
    except KeyError:
        raise ValueError(
            f"Unknown provider: {name!r} (registered: {', '.join(provider_names())})"
        ) from None


def provider_names() -> list[str]:
    return sorted(_specs)


def _build_gemini(settings: Settings, mode: str) -> ModelProvider | None:
    if not settings.has_gemini_key:
        if mode == "record":
            raise ValueError(
                "A Gemini API key is required to record: set AGENT_REPLAY_GEMINI_API_KEY"
            )
        return None
    from agent_replay.llm.gemini import GeminiProvider

    return GeminiProvider(
        settings.gemini_api_key.get_secret_value(), timeout_s=settings.timeout_s
    )


def _build_scripted(settings: Settings, mode: str) -> ModelProvider | None:
    path = os.environ.get(SCRIPT_PATH_ENV)
    if not path:
        if mode == "record":
            raise ValueError(f"{SCRIPT_PATH_ENV} is required for the scripted provider")
        return None
    from agent_replay.llm.fake import ScriptedProvider

    return ScriptedProvider.from_file(path)


def _build_claude_code(settings: Settings, mode: str) -> ModelProvider | None:
    # Found = shutil.which resolves it (a bare name on PATH, or an executable file path).
    cli = shutil.which(settings.claude_cli) if settings.claude_cli else None
    if cli is None:
        if mode == "record":
            raise ValueError(
                "The Claude Code CLI was not found: set AGENT_REPLAY_CLAUDE_CLI to the "
                "`claude` executable (a name on PATH or a path to an executable file)"
            )
        return None
    from agent_replay.llm.claude_code import ClaudeCodeProvider

    # The child runs in a temp dir, so a relative path must be made absolute here.
    cli = os.path.abspath(cli)

    return ClaudeCodeProvider(
        cli, timeout_s=settings.timeout_s, max_turns=settings.claude_max_turns
    )


register(
    ProviderSpec(
        name="gemini",
        build=_build_gemini,
        needs_credential_to_record=True,
        default_model="gemini-3.5-flash-lite",
        default_judge_model=None,
        default_prices=(0.30, 2.50),
        rate_limited=True,
    )
)
register(
    ProviderSpec(
        name="scripted",
        build=_build_scripted,
        needs_credential_to_record=False,
        # Same model as gemini so existing scripted fixtures keep their request hashes.
        default_model="gemini-3.5-flash-lite",
        default_judge_model=None,
        default_prices=(0.30, 2.50),
        rate_limited=False,
    )
)
register(
    ProviderSpec(
        name="claude_code",
        build=_build_claude_code,
        needs_credential_to_record=False,
        default_model="claude-haiku-5-5",
        default_judge_model="claude-sonnet-5-5",
        default_prices=(0.10, 0.50),
        model_prices={
            "claude-haiku-5-5": (0.10, 0.50),
            "claude-sonnet-5-5": (2.00, 10.00),
        },
        rate_limited=True,
    )
)
