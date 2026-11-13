"""JSON-safe conversion and shared validation for the API."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

UUID_PATTERN = (
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def to_json(value: Any) -> Any:
    """Recursively convert DB values: UUID/datetime to text, Decimal to a string."""
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [to_json(v) for v in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def json_list(rows: list[Any]) -> list[Any]:
    return list(to_json(rows))


def json_dict(row: dict[str, Any]) -> dict[str, Any]:
    return dict(to_json(row))
