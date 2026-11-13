"""Runtime settings read from AGENT_REPLAY_* environment variables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from agent_replay.llm.registry import DEFAULT_PROVIDER, get_provider_spec


def redact_url(url: str | None) -> str | None:
    """Replace the password in a URL with ``***``."""
    if url is None:
        return None
    parts = urlsplit(url)
    if parts.password is None:
        return url
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parts.port is not None:
        host += f":{parts.port}"
    user = parts.username or ""
    return urlunsplit(parts._replace(netloc=f"{user}:***@{host}"))


class Settings(BaseSettings):
    """Process settings. Environment only: no .env file is ever read."""

    model_config = SettingsConfigDict(env_prefix="AGENT_REPLAY_", extra="forbid")

    database_url: str | None = Field(None, repr=False)
    gemini_api_key: SecretStr = SecretStr("")
    provider: str = DEFAULT_PROVIDER
    # No defaults here: _resolve_provider_defaults fills model, judge_model and the prices
    # from the provider's registered spec when the environment leaves them unset.
    model: str
    judge_model: str
    rpm: int = Field(60, gt=0)
    rpd: int = Field(5000, gt=0)
    run_budget_usd: float = Field(1.50, ge=0)
    price_input_per_m: float = Field(ge=0)
    price_output_per_m: float = Field(ge=0)
    judge_price_input_per_m: float = Field(ge=0)
    judge_price_output_per_m: float = Field(ge=0)
    timeout_s: float = Field(30, gt=0)
    max_attempts: int = Field(3, ge=1)
    backoff_base_s: float = Field(2, ge=0)
    max_steps: int = Field(6, ge=1)
    tool_timeout_s: float = Field(10, gt=0)
    seed: int = 7
    temperature: float = 0
    cors_origins: list[str] = Field(default_factory=list)
    # Host header values the API answers (DNS-rebinding guard). Comma-separated or a
    # JSON list in AGENT_REPLAY_ALLOWED_HOSTS.
    allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1"]
    )
    viewer_dist: Path | None = None
    # claude_code provider: the CLI (a name on PATH or a path) and an opt-in --max-turns.
    claude_cli: str = "claude"
    claude_max_turns: int | None = Field(None, ge=1)

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text.startswith("["):
            return json.loads(text)
        return [h.strip() for h in text.split(",") if h.strip()]

    @model_validator(mode="before")
    @classmethod
    def _resolve_provider_defaults(cls, data: Any) -> Any:
        """Fill model, judge_model and prices the environment left unset.

        Order: explicit value, then the provider's registered default. An unregistered
        provider name is not an error here (the factory rejects it); it takes the default
        provider's values.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        try:
            spec = get_provider_spec(str(data.get("provider") or DEFAULT_PROVIDER))
        except ValueError:
            spec = get_provider_spec(DEFAULT_PROVIDER)
        if data.get("model") is None:
            data["model"] = spec.default_model
        if data.get("judge_model") is None:
            data["judge_model"] = spec.default_judge_model or data["model"]
        table = spec.model_prices.get(str(data["model"]), spec.default_prices)
        if data.get("price_input_per_m") is None:
            data["price_input_per_m"] = table[0]
        if data.get("price_output_per_m") is None:
            data["price_output_per_m"] = table[1]
        # Judge prices: the agent prices when the judge runs the agent's model, else the
        # judge model's registered prices, else the provider's default prices.
        if data["judge_model"] == data["model"]:
            judge = (data["price_input_per_m"], data["price_output_per_m"])
        else:
            judge = spec.model_prices.get(str(data["judge_model"]), spec.default_prices)
        if data.get("judge_price_input_per_m") is None:
            data["judge_price_input_per_m"] = judge[0]
        if data.get("judge_price_output_per_m") is None:
            data["judge_price_output_per_m"] = judge[1]
        return data

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key.get_secret_value().strip())

    def safe_dict(self) -> dict[str, Any]:
        """Settings as a dict with the API key replaced by a presence flag."""
        data = self.model_dump(mode="json", exclude={"gemini_api_key"})
        data["database_url"] = redact_url(self.database_url)
        data["gemini_api_key"] = "set" if self.has_gemini_key else "unset"
        return data
