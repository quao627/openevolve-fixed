#!/usr/bin/env bash
# Run all math examples with OpenEvolve using Claude Opus 4.6
# Usage: ./run_all_math.sh [problem_name]
#   Without args: runs all 17 math problems sequentially
#   With arg: runs only the specified problem (e.g., ./run_all_math.sh circle_packing)
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
    circle_packing
    circle_packing_rect
    erdos_min_overlap
    first_autocorr_ineq
    heilbronn_convex_13
    heilbronn_convex_14
    heilbronn_triangle
    hexagon_packing_11
    hexagon_packing_12
    matmul
    minimizing_max_min_dist_2d
    minimizing_max_min_dist_3d
    second_autocorr_ineq
    signal_processing
    sums_diffs_finite_sets
    third_autocorr_ineq
    uncertainty_ineq
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

echo "All math problems complete!"
