#!/usr/bin/env bash
set -a
source .env
set +a

EXAMPLE_DIR="examples/perf_takehome"
OUTPUT_DIR="${EXAMPLE_DIR}/openevolve_output"

python openevolve-run.py \
  "${EXAMPLE_DIR}/initial_program.py" \
  "${EXAMPLE_DIR}/evaluator.py" \
  --config "${EXAMPLE_DIR}/config_opus.yaml" \
  --output "$OUTPUT_DIR" \
  --iterations 300 \
  2>&1 | tee "${EXAMPLE_DIR}/evolution.log"
