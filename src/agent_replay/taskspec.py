"""Task specs loaded from a JSONL file, one task per line."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ValidationError

from agent_replay import jsonio


class TaskSpec(BaseModel):
    id: str
    prompt: str
    category: str
    expected_text: list[str]
    must_call_tools: list[str]
    max_steps: int
    rubric: str


def load_tasks(path: str | Path) -> list[TaskSpec]:
    """Read tasks in file order; errors name the file, line and field, never the value."""
    p = Path(path)
    tasks: list[TaskSpec] = []
    seen: dict[str, int] = {}
    for lineno, line in jsonio.jsonl_lines(p.read_text(encoding="utf-8")):
        try:
            data = jsonio.loads(line)
        except ValueError:
            raise ValueError(
                f"{p.name}:{lineno}: invalid task (not valid JSON)"
            ) from None
        try:
            task = TaskSpec.model_validate(data)
        except ValidationError as exc:
            # Only the error type and field: the input may not be a task file at all.
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first["loc"]) or "line"
            raise ValueError(
                f"{p.name}:{lineno}: invalid task ({first['type']} at {where})"
            ) from None
        if task.id in seen:
            raise ValueError(
                f"{p.name}:{lineno}: duplicate task id {task.id!r} "
                f"(first seen on line {seen[task.id]})"
            )
        seen[task.id] = lineno
        tasks.append(task)
    return tasks


def get_task(tasks: list[TaskSpec], task_id: str) -> TaskSpec:
    for t in tasks:
        if t.id == task_id:
            return t
    raise KeyError(f"task not found: {task_id}")
