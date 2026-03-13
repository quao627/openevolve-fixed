#!/usr/bin/env bash
set -a
source .env
set +a

EXAMPLE_DIR="examples/circle_packing"
OUTPUT_DIR="${EXAMPLE_DIR}/openevolve_output_test"

# Phase 1: Initial exploration
python openevolve-run.py \
  "${EXAMPLE_DIR}/initial_program.py" \
  "${EXAMPLE_DIR}/evaluator.py" \
  --config "${EXAMPLE_DIR}/config_phase_1.yaml" \
  --output "$OUTPUT_DIR" \
  2>&1 | tee "${EXAMPLE_DIR}/evolution.log"

# Find latest checkpoint
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoints/checkpoint_* 2>/dev/null \
  | sort -V | tail -1)

if [ -z "$LATEST_CKPT" ]; then
  echo "ERROR: No checkpoint found after phase 1"
  exit 1
fi

# Phase 2: Break through the plateau
python openevolve-run.py \
  "${LATEST_CKPT}/best_program.py" \
  "${EXAMPLE_DIR}/evaluator.py" \
  --config "${EXAMPLE_DIR}/config_phase_2.yaml" \
  --checkpoint "$LATEST_CKPT" \
  --output "$OUTPUT_DIR" \
  2>&1 | tee -a "${EXAMPLE_DIR}/evolution.log"