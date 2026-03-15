#!/bin/bash
# Run TriMul kernel engineering evolution with Claude Opus 4.6
# Requires: ANTHROPIC_API_KEY, GPU with torch+triton installed

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "$SCRIPT_DIR/../../openevolve-run.py" \
    "$SCRIPT_DIR/initial_program.py" \
    "$SCRIPT_DIR/evaluator.py" \
    --config "$SCRIPT_DIR/config.yaml" \
    --iterations 200
