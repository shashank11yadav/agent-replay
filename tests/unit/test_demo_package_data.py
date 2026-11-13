from importlib import resources

from agent_replay.demo.agent import load_prompt


def test_demo_resources_load_via_importlib() -> None:
    root = resources.files("agent_replay.demo")
    for name in ("orders.json", "products.json", "rates.json"):
        assert root.joinpath("data", name).read_text("utf-8").strip()
    assert root.joinpath("example", "task.jsonl").is_file()
    assert load_prompt("minimal").strip() and load_prompt("structured").strip()
