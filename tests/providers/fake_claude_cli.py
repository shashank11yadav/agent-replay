"""Fake `claude` CLI for the claude_code provider tests. Never talks to anything; driven by a scenario file.

Run through a tiny launcher script (see claude_cli_harness.py), not directly. Behaviour comes from the
JSON file named by FAKE_CLAUDE_SCENARIO (an env var that does not start with AGENT_REPLAY_, so
the provider's environment stripping leaves it alone)::

    {"calls": [<step>, <step>, ...]}      # call N uses step N (the last one repeats)

A step may set: ``stdout`` (raw text) or ``stdout_json`` (object, dumped), ``stderr``,
``exit_code``, ``sleep`` (seconds, before answering), ``kill_signal`` (the process kills itself
with it, after the sleep), ``write_file_in_cwd`` (leaves a file in the working directory).

Every call appends one JSON line to ``<scenario>.calls.jsonl`` with argv, stdin (base64), cwd,
the cwd listing and the full environment.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time


def main() -> int:
    scenario_path = os.environ["FAKE_CLAUDE_SCENARIO"]
    with open(scenario_path, encoding="utf-8") as fh:
        steps = json.load(fh)["calls"]
    log_path = scenario_path + ".calls.jsonl"
    index = 0
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as fh:
            index = sum(1 for _ in fh)
    step = steps[min(index, len(steps) - 1)]

    stdin = sys.stdin.buffer.read()
    record = {
        "argv0": globals().get("LAUNCHER_ARGV0", sys.argv[0]),
        "argv": sys.argv[1:],
        "stdin_b64": base64.b64encode(stdin).decode("ascii"),
        "cwd": os.getcwd(),
        "cwd_listing": sorted(os.listdir(os.getcwd())),
        "env": dict(os.environ),
    }
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    if step.get("write_file_in_cwd"):
        with open(os.path.join(os.getcwd(), "junk.txt"), "w", encoding="utf-8") as fh:
            fh.write("left behind")
    if step.get("sleep"):
        time.sleep(float(step["sleep"]))
    if step.get("kill_signal"):
        os.kill(os.getpid(), int(step["kill_signal"]))
        time.sleep(30)
    if "stdout_json" in step:
        sys.stdout.write(json.dumps(step["stdout_json"]))
    elif "stdout" in step:
        sys.stdout.write(step["stdout"])
    sys.stdout.flush()
    if step.get("stderr"):
        sys.stderr.write(step["stderr"])
        sys.stderr.flush()
    return int(step.get("exit_code", 0))


if __name__ == "__main__":
    sys.exit(main())
