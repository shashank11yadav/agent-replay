"""Run and judge codes shared by the runner, the suite loop, the gate and fixtures verify.

A stop code means the provider or the run budget failed, not the agent: the run is stored with
the code as its ``error_type``, the suite stops, and the gate reports ERROR instead of judging it.
"""

from __future__ import annotations

RATE_LIMIT = "rate_limit"
QUOTA_EXHAUSTED = "quota_exhausted"
BUDGET_EXCEEDED = "budget_exceeded"
TRANSIENT_ERROR = "transient_error"

STOP_CODES = frozenset({RATE_LIMIT, QUOTA_EXHAUSTED, BUDGET_EXCEEDED, TRANSIENT_ERROR})

# A ``rate_limit:<n>`` fault outlasted the client's retries. The 429s were injected, so the
# provider is fine: this is an ordinary run error, not a stop code, and a fault suite goes on
# to the next task.
INJECTED_RATE_LIMIT = "injected_rate_limit"

# Judge verdicts that stop scoring with an error exit.
JUDGE_STOP_CODES = frozenset({QUOTA_EXHAUSTED, BUDGET_EXCEEDED})

# A judge row with this verdict and any other ``error`` failed for provider reasons.
JUDGE_ERROR = "judge_error"
# The judge call missed the cache and no live model was available.
JUDGE_SKIPPED = "judge_skipped"
UNPARSEABLE_VERDICT = "unparseable verdict"
