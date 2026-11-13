"""Strict JSON reading shared by task files, fixture bundles and gate files.

Files are read as UTF-8 whatever the locale, and NaN or Infinity is refused: neither is
valid JSON, and a NaN threshold or score would compare false against everything.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name}")


def loads(text: str) -> Any:
    """``json.loads`` that raises ValueError on NaN and Infinity anywhere in the document."""
    return json.loads(text, parse_constant=_reject_constant)


def read_json(path: str | Path) -> Any:
    """Parse a whole JSON file. Raises OSError or ValueError."""
    return loads(Path(path).read_text(encoding="utf-8"))


def jsonl_lines(text: str) -> Iterator[tuple[int, str]]:
    """The non-blank lines of a JSONL document with their 1-based line numbers."""
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            yield lineno, line
