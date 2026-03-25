#!/usr/bin/env bash
# Run all ADRS examples with OpenEvolve using Claude Opus 4.6
# Usage: ./run_all_adrs.sh [problem_name]
#   Without args: runs all 5 ADRS problems sequentially
#   With arg: runs only the specified problem (e.g., ./run_all_adrs.sh prism)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENEVOLVE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$OPENEVOLVE_DIR/openevolve-run.py"

# Load .env if present (for ANTHROPIC_API_KEY)
if [ -f "$OPENEVOLVE_DIR/.env" ]; then
    set -a; source "$OPENEVOLVE_DIR/.env"; set +a
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set. Export it or add to $OPENEVOLVE_DIR/.env"
    exit 1
fi

PROBLEMS=(
    cloudcast
    eplb
    llm_sql
    prism
    txn_scheduling
)

# Filter to single problem if specified
if [ $# -ge 1 ]; then
    PROBLEMS=("$1")
fi

for problem in "${PROBLEMS[@]}"; do
    PROBLEM_DIR="$SCRIPT_DIR/$problem"
    if [ ! -d "$PROBLEM_DIR" ]; then
        echo "SKIP: $problem (directory not found)"
        continue
    fi

    echo "========================================"
    echo "Running: $problem"
    echo "========================================"

    OUTPUT_DIR="$PROBLEM_DIR/openevolve_output"
    mkdir -p "$OUTPUT_DIR"

    python "$RUNNER" \
        "$PROBLEM_DIR/initial_program.py" \
        "$PROBLEM_DIR/evaluator.py" \
        --config "$PROBLEM_DIR/config.yaml" \
        --output "$OUTPUT_DIR" \
        --iterations 100 \
        2>&1 | tee "$PROBLEM_DIR/evolution.log"

    echo "Finished: $problem"
    echo ""
done

echo "All ADRS problems complete!"
