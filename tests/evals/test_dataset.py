"""Eval dataset, prompts, gate thresholds and suite config.

Expected values are recomputed here by calling the real demo tool functions on the
bundled data; nothing is copied from the task labels.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from importlib import resources

import pytest
from support import ROOT

from agent_replay.config import Settings
from agent_replay.demo.agent import available_variants, load_prompt
from agent_replay.demo.tools import TOOLS
from agent_replay.scoring.checks import expected_text as _expected_text_check
from agent_replay.taskspec import TaskSpec, load_tasks

TASKS_FILE = ROOT / "evals" / "tasks" / "tasks.jsonl"
GATE_FILE = ROOT / "evals" / "gate.json"
SUITE_FILE = ROOT / "evals" / "suite.json"
DEMO = ROOT / "src" / "agent_replay" / "demo"
DATA = DEMO / "data"
PROMPTS = DEMO / "prompts"

ORDER_ID = re.compile(r"\b[A-Z]\d{4}\b")
ANY_ID = re.compile(r"\b[A-Z]\d{3,5}\b")


def _data(name: str):
    return json.loads((DATA / name).read_text("utf-8"))


def _tasks() -> list[TaskSpec]:
    return load_tasks(TASKS_FILE)


def _by_category(cat: str) -> list[TaskSpec]:
    return [t for t in _tasks() if t.category == cat]


# 1
def test_tasks_load_30_unique_ids() -> None:
    tasks = _tasks()
    assert len(tasks) == 30
    assert len({t.id for t in tasks}) == 30


# 2
def test_category_counts() -> None:
    counts = Counter(t.category for t in _tasks())
    assert dict(counts) == {
        "lookup": 10,
        "multi_tool": 10,
        "search": 5,
        "not_found": 5,
    }


# spec: 20 orders, 15 products, 4 currencies with fixed rates
def test_data_sizes_match_spec() -> None:
    assert len(_data("orders.json")) == 20
    assert len(_data("products.json")) == 15
    assert len(_data("rates.json")["rates"]) == 4


# 3
def _corpus(task: TaskSpec) -> tuple[str, str]:
    """Return (all tool output text, convert_currency output text) for the task,
    produced by calling only the tools the task lists, with arguments derived
    from the prompt and the data."""
    orders = {o["order_id"]: o for o in _data("orders.json")}
    products = _data("products.json")
    currencies = list(_data("rates.json")["rates"])
    listed = set(task.must_call_tools)
    assert listed, f"{task.id}: must_call_tools empty"
    assert listed <= set(TOOLS), f"{task.id}: unknown tools {listed - set(TOOLS)}"

    outputs: list[object] = []
    converts: list[object] = []
    ids = ORDER_ID.findall(task.prompt)
    if "get_order" in listed:
        for oid in ids:
            outputs.append(TOOLS["get_order"](oid))
    if "search_products" in listed:
        lowered = task.prompt.lower()
        queries: set[str] = set()
        for p in products:
            if p["name"].lower() in lowered:
                queries.add(p["name"])
            for word in p["name"].split():
                if len(word) > 3 and re.search(
                    rf"\b{re.escape(word.lower())}", lowered
                ):
                    queries.add(word)
        for q in sorted(queries):
            outputs.append(TOOLS["search_products"](q))
    if "convert_currency" in listed:
        for oid in ids:
            if oid in orders:
                o = orders[oid]
                for cur in currencies:
                    if cur != o["currency"]:
                        converts.append(
                            TOOLS["convert_currency"](o["total"], o["currency"], cur)
                        )
        for p in products:
            if p["name"].lower() in task.prompt.lower():
                for cur in currencies:
                    if cur != p["currency"]:
                        converts.append(
                            TOOLS["convert_currency"](p["price"], p["currency"], cur)
                        )
    text = json.dumps(outputs + converts, ensure_ascii=False).lower()
    ctext = json.dumps(converts, ensure_ascii=False).lower()
    return text, ctext


@pytest.mark.parametrize(
    "task", _tasks() if TASKS_FILE.exists() else [], ids=lambda t: t.id
)
def test_expected_text_reproduced_by_tools(task: TaskSpec) -> None:
    assert task.expected_text, f"{task.id}: expected_text empty"
    if task.category == "not_found":
        return  # covered by the not_found test below
    text, ctext = _corpus(task)
    assert text != "[]", f"{task.id}: no tool output derivable from the prompt"
    for value in task.expected_text:
        v = value.lower()
        assert v in text or v.replace(",", "") in text, (
            f"{task.id}: {value!r} not produced by {task.must_call_tools}"
        )
    if task.category == "multi_tool":
        assert "convert_currency" in task.must_call_tools
        assert any(v.lower().replace(",", "") in ctext for v in task.expected_text), (
            f"{task.id}: no answer value comes from convert_currency"
        )


def test_tasks_file_exists_for_line_3() -> None:
    assert TASKS_FILE.exists()
    assert len(_tasks()) == 30


# 4
def test_not_found_tasks_reference_missing_ids() -> None:
    tasks = _by_category("not_found")
    assert len(tasks) == 5
    order_ids = {o["order_id"] for o in _data("orders.json")}
    product_ids = {p["id"] for p in _data("products.json")}
    for t in tasks:
        ids = ANY_ID.findall(t.prompt)
        assert ids, f"{t.id}: no order or product id in prompt"
        for i in ids:
            assert i not in order_ids and i not in product_ids, (
                f"{t.id}: {i} exists in the data"
            )
        for i in ORDER_ID.findall(t.prompt):
            assert TOOLS["get_order"](i) == {"error": "not_found"}


# 5
def test_rubric_and_max_steps() -> None:
    limit = Settings().max_steps
    tasks = _tasks()
    assert len(tasks) == 30
    for t in tasks:
        assert t.rubric.strip(), f"{t.id}: empty rubric"
        assert 1 <= t.max_steps <= limit, f"{t.id}: max_steps {t.max_steps}"


# 6
def test_gate_thresholds() -> None:
    gate = json.loads(GATE_FILE.read_text("utf-8"))
    assert gate["max_pass_rate_drop_pp"] == 5
    assert gate["max_token_increase_pct"] == 15
    assert gate["max_p95_increase_pct"] == 20


# 7
def test_suite_names_shipped_variant_with_prompt() -> None:
    suite = json.loads(SUITE_FILE.read_text("utf-8"))
    name = suite.get("variant", suite.get("shipped_variant"))
    assert isinstance(name, str) and name, "suite.json needs a 'variant' string"
    assert name in available_variants()
    assert (PROMPTS / f"{name}.txt").is_file()


# 8
def test_load_prompt_all_variants() -> None:
    for v in ("minimal", "structured", "plan_act", "structured_regressed"):
        assert v in available_variants()
        assert load_prompt(v).strip(), v
    assert load_prompt("structured_regressed") != load_prompt("structured")
    assert load_prompt("plan_act") != load_prompt("minimal")


# 9
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}")
REAL_NAMES = [
    "elon musk", "taylor swift", "barack obama", "donald trump", "joe biden",
    "bill gates", "jeff bezos", "mark zuckerberg", "tom hanks", "oprah winfrey",
    "john smith", "jane doe", "michael jordan", "lebron james", "emma watson",
    "brad pitt", "angelina jolie", "kanye west", "steve jobs", "tim cook",
]  # fmt: skip


def test_no_pii_in_tasks_prompts_or_data() -> None:
    files = [TASKS_FILE, *sorted(PROMPTS.glob("*.txt")), *sorted(DATA.glob("*.json"))]
    assert TASKS_FILE.exists()
    assert len(list(PROMPTS.glob("*.txt"))) >= 4
    for f in files:
        text = f.read_text("utf-8")
        assert not EMAIL.search(text), f"email in {f.name}"
        assert not PHONE.search(text), f"phone in {f.name}"
        low = text.lower()
        for name in REAL_NAMES:
            assert name not in low, f"{name!r} in {f.name}"
    assert resources.files("agent_replay.demo").joinpath("data").is_dir()


# ---------------------------------------------------------------------------
# Fix-cycle additions: labels must score real answers correctly.
# ---------------------------------------------------------------------------
NUMERIC = re.compile(r"^\d+(\.\d+)?$")


def _numbers(obj: object) -> list[float]:
    """All money values in tool output."""
    out: list[float] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("price", "total", "converted") and isinstance(v, (int, float)):
                out.append(float(v))
            else:
                out.extend(_numbers(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_numbers(v))
    return out


def _true_values(task: TaskSpec) -> list[float]:
    """True numeric values the listed tools produce for the task."""
    orders = {o["order_id"]: o for o in _data("orders.json")}
    products = _data("products.json")
    currencies = list(_data("rates.json")["rates"])
    vals: list[float] = []
    for oid in ORDER_ID.findall(task.prompt):
        if oid in orders and "convert_currency" in task.must_call_tools:
            o = orders[oid]
            for cur in currencies:
                if cur != o["currency"]:
                    vals += _numbers(
                        TOOLS["convert_currency"](o["total"], o["currency"], cur)
                    )
    low = task.prompt.lower()
    for p in products:
        if p["name"].lower() in low:
            vals += _numbers(TOOLS["search_products"](p["name"]))
    return vals


def _renderings(v: float) -> list[str]:
    r = [f"{v:.2f}", f"{v:,.2f}"]
    if abs(v - round(v)) < 1e-9:
        r += [str(round(v)), f"{round(v):,}"]
    return r


def _numeric_tasks() -> list[TaskSpec]:
    if not TASKS_FILE.exists():
        return []
    return [
        t
        for t in _tasks()
        if t.category != "not_found" and any(NUMERIC.match(x) for x in t.expected_text)
    ]


@pytest.mark.parametrize("task", _numeric_tasks(), ids=lambda t: t.id)
def test_numeric_label_is_format_robust(task: TaskSpec) -> None:
    truths = _true_values(task)
    assert truths, f"{task.id}: no tool values derivable"
    for label in (x for x in task.expected_text if NUMERIC.match(x)):
        match = [v for v in truths if abs(v - float(label)) < 0.005]
        assert match, f"{task.id}: {label!r} is not a tool value {truths}"
        for rendering in _renderings(match[0]):
            answer = f"The value is {rendering} in total."
            res = _expected_text_check(task, {"final_output": answer}, [])
            assert res.passed, (
                f"{task.id}: label {label!r} fails on correct rendering {rendering!r}"
            )


@pytest.mark.parametrize(
    "task", _tasks() if TASKS_FILE.exists() else [], ids=lambda t: t.id
)
def test_labels_not_in_own_prompt(task: TaskSpec) -> None:
    prompt = task.prompt.lower()
    for label in task.expected_text:
        assert label.lower() not in prompt, (
            f"{task.id}: label {label!r} is echoed by the prompt"
        )


@pytest.mark.parametrize("task", _numeric_tasks(), ids=lambda t: t.id)
def test_numeric_label_not_inside_other_values(task: TaskSpec) -> None:
    truths = _true_values(task)
    others: list[float] = [float(p["price"]) for p in _data("products.json")]
    others += [float(o["total"]) for o in _data("orders.json")]
    for label in (x for x in task.expected_text if NUMERIC.match(x)):
        true_v = [v for v in truths if abs(v - float(label)) < 0.005]
        for other in others:
            if any(abs(other - t) < 0.005 for t in true_v):
                continue
            for r in (f"{other:.2f}", f"{other:,.2f}"):
                assert label not in r, (
                    f"{task.id}: label {label!r} is inside {r!r}, a different value"
                )


def test_not_found_label_matches_shipped_prompt_wording() -> None:
    suite = json.loads(SUITE_FILE.read_text("utf-8"))
    prompt = load_prompt(suite["variant"]).lower()
    tasks = _by_category("not_found")
    assert len(tasks) == 5
    for t in tasks:
        for label in t.expected_text:
            assert label.lower() in prompt, (
                f"{t.id}: {label!r} is not wording the {suite['variant']} prompt "
                "tells the model to use"
            )


def test_order_totals_consistent_with_items() -> None:
    products = {p["id"]: p for p in _data("products.json")}
    rates = _data("rates.json")["rates"]
    for o in _data("orders.json"):
        total = 0.0
        for item in o["items"]:
            assert item in products, f"{o['order_id']}: unknown item {item}"
            p = products[item]
            total += p["price"] * rates[o["currency"]] / rates[p["currency"]]
        assert abs(total - o["total"]) <= 0.01, (
            f"{o['order_id']}: total {o['total']} {o['currency']} != items {total:.2f}"
        )
