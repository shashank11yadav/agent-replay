"""Per-process run budget: cost estimate, shared spend tracker and pre-call guard."""

from __future__ import annotations

from collections.abc import Mapping

from agent_replay.llm.provider import ModelProvider
from agent_replay.llm.types import BudgetExceeded, ModelRequest, ModelResponse

DEFAULT_PRICE_INPUT_PER_M = 0.30
DEFAULT_PRICE_OUTPUT_PER_M = 2.50


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    price_input_per_m: float,
    price_output_per_m: float,
) -> float:
    return (
        input_tokens * price_input_per_m / 1e6
        + output_tokens * price_output_per_m / 1e6
    )


class BudgetTracker:
    """Cumulative estimated spend for this process, shared by agent and judge calls."""

    def __init__(
        self,
        price_input_per_m: float = DEFAULT_PRICE_INPUT_PER_M,
        price_output_per_m: float = DEFAULT_PRICE_OUTPUT_PER_M,
        *,
        model_prices: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        self.model_prices: dict[str, tuple[float, float]] = dict(model_prices or {})
        self.spent_usd = 0.0
        self.largest_call_usd = 0.0

    def cost_of(self, response: ModelResponse, model: str | None = None) -> float:
        """Estimated cost of a response at the model's prices; cache hits cost 0.

        A model that is None or not in the table is priced at the default pair.
        """
        if response.cache_status == "hit":
            return 0.0
        price_in, price_out = self.price_input_per_m, self.price_output_per_m
        if model is not None and model in self.model_prices:
            price_in, price_out = self.model_prices[model]
        return estimate_cost_usd(
            response.input_tokens, response.output_tokens, price_in, price_out
        )

    def record(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd
        self.largest_call_usd = max(self.largest_call_usd, cost_usd)


class BudgetGuardProvider:
    """Refuses a call when spent >= budget or spent + largest call so far > budget."""

    def __init__(
        self,
        provider: ModelProvider,
        budget_usd: float,
        tracker: BudgetTracker | None = None,
    ) -> None:
        self._provider = provider
        self._budget_usd = budget_usd
        self.tracker = tracker if tracker is not None else BudgetTracker()

    def generate(self, request: ModelRequest) -> ModelResponse:
        spent = self.tracker.spent_usd
        projected = self.tracker.largest_call_usd
        if spent >= self._budget_usd or spent + projected > self._budget_usd:
            raise BudgetExceeded(spent, self._budget_usd)
        response = self._provider.generate(request)
        self.tracker.record(self.tracker.cost_of(response, request.model))
        return response
