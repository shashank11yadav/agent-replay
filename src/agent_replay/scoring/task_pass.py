"""The one task-pass rule, shared by the run diff, suite metrics, the gate and the API.

The viewer has no copy of the rule: it shows the ``passed`` value the API computes here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def task_passed(scores: Iterable[Mapping[str, Any]]) -> bool | None:
    """True when every deterministic check passed and the judge passed.

    None when the run is not fully scored (no deterministic row or no judge row), so a run
    scored with ``--no-judge`` is neither a pass nor a fail. A judge row with a null verdict
    (judge error or skipped) is a fail.
    """
    rows = list(scores)
    checks = [s for s in rows if s.get("kind") != "judge"]
    judges = [s for s in rows if s.get("kind") == "judge"]
    if not checks or not judges:
        return None
    return all(s.get("passed") is True for s in checks + judges)
