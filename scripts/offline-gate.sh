#!/usr/bin/env bash
# Run the regression gate offline on the committed fixtures and check both verdicts:
# the baseline replayed from its own cache must PASS, and the committed prompt regression
# must FAIL. Exits 0 only when both verdicts are as expected.
#
# AGENT_REPLAY_DATABASE_URL names a Postgres server and a database the role can use to
# create databases. The script creates a throwaway database next to it, works only there
# and drops it on exit.
#
# No live model is ever called: the Claude Code CLI and the Gemini key are blanked, so a
# cache miss is SKIPPED instead of being sent anywhere.
set -euo pipefail

cd "$(dirname "$0")/.."

: "${AGENT_REPLAY_DATABASE_URL:?set AGENT_REPLAY_DATABASE_URL (e.g. postgresql://postgres:postgres@127.0.0.1:5432/postgres)}"
ADMIN_URL="$AGENT_REPLAY_DATABASE_URL"

# Only the provider and models from the baseline manifest apply; drop any local overrides.
for var in $(env | sed -n 's/^\(AGENT_REPLAY_[A-Z0-9_]*\)=.*/\1/p'); do
    unset "$var"
done
export AGENT_REPLAY_CLAUDE_CLI=""
export AGENT_REPLAY_GEMINI_API_KEY=""

ar() { uv run --quiet agent-replay "$@"; }
py() { uv run --quiet python - "$@"; }
manifest() { py "$1" "$2" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2]) or "")
EOF
}

DB_NAME="agent_replay_gate_$(date +%s)_$$"
echo "throwaway database: $DB_NAME" >&2

db_admin() { py "$ADMIN_URL" "$1" "$DB_NAME" <<'EOF'
import sys
import psycopg
from psycopg import sql
url, action, name = sys.argv[1:4]
with psycopg.connect(url, autocommit=True) as conn:
    if action == "create":
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    else:
        conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
        )
EOF
}

url_for() { py "$ADMIN_URL" "$DB_NAME" <<'EOF'
import sys
from urllib.parse import urlsplit, urlunsplit
url, name = sys.argv[1:3]
print(urlunsplit(urlsplit(url)._replace(path="/" + name)))
EOF
}

# The drop uses IF EXISTS, so cleanup is safe before the database has been created.
cleanup() {
    db_admin drop >/dev/null 2>&1 || echo "warning: could not drop database $DB_NAME" >&2
    rm -rf "$WORK"
}

WORK="$(mktemp -d "${TMPDIR:-/tmp}/agent-replay-gate.XXXXXX")"
trap cleanup EXIT
db_admin create
AGENT_REPLAY_DATABASE_URL="$(url_for)"
export AGENT_REPLAY_DATABASE_URL

PROVIDER="$(manifest evals/baseline/manifest.json provider)"
AGENT_MODEL="$(manifest evals/baseline/manifest.json agent_model)"
JUDGE_MODEL="$(manifest evals/baseline/manifest.json judge_model)"
[ -n "$PROVIDER" ] && export AGENT_REPLAY_PROVIDER="$PROVIDER"
[ -n "$AGENT_MODEL" ] && export AGENT_REPLAY_MODEL="$AGENT_MODEL"
[ -n "$JUDGE_MODEL" ] && export AGENT_REPLAY_JUDGE_MODEL="$JUDGE_MODEL"

ar db migrate >/dev/null
ar fixtures import evals/baseline >/dev/null
BASE="$(manifest evals/baseline/manifest.json suite_run_id)"

# Captured first: under pipefail a failing replay in a pipeline would end the script silently.
if ! REPLAY_OUT="$(ar suite replay --source "$BASE" --tasks evals/tasks/tasks.jsonl --label offline-check)"; then
    echo "baseline replay:   ERROR (replay stopped)"
    exit 1
fi
OFFLINE="$(printf '%s\n' "$REPLAY_OUT" | tail -n 1)"
ar suite score --suite-run "$OFFLINE" --tasks evals/tasks/tasks.jsonl >/dev/null

# gate-check prints "verdict: X" and exits 1 on FAIL or ERROR.
verdict() {
    ar gate --baseline "$BASE" --candidate "$1" --thresholds evals/gate.json \
        --out-dir "$WORK/$2" >/dev/null
    ar gate-check "$WORK/$2/verdict.json" | sed -n 's/^verdict: //p' || true
}

BASELINE_VERDICT="$(verdict "$OFFLINE" baseline)"

ar fixtures import evals/regression >/dev/null
REG="$(manifest evals/regression/manifest.json suite_run_id)"
REGRESSION_VERDICT="$(verdict "$REG" regression)"

echo "baseline replay:   $BASELINE_VERDICT (expected PASS)"
echo "prompt regression: $REGRESSION_VERDICT (expected FAIL)"
[ "$BASELINE_VERDICT" = "PASS" ] && [ "$REGRESSION_VERDICT" = "FAIL" ]
