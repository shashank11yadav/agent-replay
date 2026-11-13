"""Three synthetic tools over the bundled demo data."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import cache
from importlib import resources
from typing import Any

from pydantic import BaseModel, Field

from agent_replay.llm import ToolSchema


@cache
def _load(name: str) -> Any:
    return json.loads(
        resources.files("agent_replay.demo").joinpath("data", name).read_text("utf-8")
    )


def get_order(order_id: str) -> dict[str, Any]:
    for order in _load("orders.json"):
        if order["order_id"] == order_id:
            return dict(order)
    return {"error": "not_found"}


def search_products(query: str) -> dict[str, Any]:
    needle = query.lower()
    found = [dict(p) for p in _load("products.json") if needle in p["name"].lower()]
    found.sort(key=lambda p: p["id"])
    return {"products": found}


def convert_currency(
    amount: float, from_currency: str, to_currency: str
) -> dict[str, Any]:
    rates: dict[str, float] = _load("rates.json")["rates"]
    if from_currency not in rates or to_currency not in rates:
        return {"error": "unknown_currency"}
    rate = rates[to_currency] / rates[from_currency]
    return {
        "from": from_currency,
        "to": to_currency,
        "converted": round(amount * rate, 2),
    }


class OrderOutput(BaseModel):
    order_id: str
    customer: str
    status: str
    items: list[str]
    total: float
    currency: str


class ProductOutput(BaseModel):
    id: str
    name: str
    price: float
    currency: str


class SearchOutput(BaseModel):
    products: list[ProductOutput]


class ConvertOutput(BaseModel):
    from_: str = Field(alias="from")
    to: str
    converted: float


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_order": get_order,
    "search_products": search_products,
    "convert_currency": convert_currency,
}

OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "get_order": OrderOutput,
    "search_products": SearchOutput,
    "convert_currency": ConvertOutput,
}

TOOL_SCHEMAS: dict[str, ToolSchema] = {
    "get_order": ToolSchema(
        name="get_order",
        description="Look up an order by its id (for example A1001).",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order id"}},
            "required": ["order_id"],
        },
    ),
    "search_products": ToolSchema(
        name="search_products",
        description="Find products whose name contains the query, case-insensitively.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Name fragment"}},
            "required": ["query"],
        },
    ),
    "convert_currency": ToolSchema(
        name="convert_currency",
        description="Convert an amount between currencies (USD, EUR, GBP, JPY) at a fixed rate.",
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    ),
}
