#!/usr/bin/env bash
# Run this problem with OpenEvolve using Claude Opus 4.6
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENEVOLVE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNNER="$OPENEVOLVE_DIR/openevolve-run.py"

if [ -f "$OPENEVOLVE_DIR/.env" ]; then
    set -a; source "$OPENEVOLVE_DIR/.env"; set +a
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set"
    exit 1
fi

OUTPUT_DIR="$SCRIPT_DIR/openevolve_output"
mkdir -p "$OUTPUT_DIR"

python "$RUNNER" \
    "$SCRIPT_DIR/initial_program.py" \
    "$SCRIPT_DIR/evaluator.py" \
    --config "$SCRIPT_DIR/config.yaml" \
    --output "$OUTPUT_DIR" \
    --iterations 100 \
    2>&1 | tee "$SCRIPT_DIR/evolution.log"
