#!/usr/bin/env bash
#
# Two-phase evolution for all AlphaEvolve problems using Qwen-2.5-7B via vLLM.
#
# Usage:
#   ./scripts/run_all_alphaevolve.sh --serve          # terminal 1: start vLLM
#   ./scripts/run_all_alphaevolve.sh                   # terminal 2: run benchmark
#   ./scripts/run_all_alphaevolve.sh --problem matmul  # single problem
#   ./scripts/run_all_alphaevolve.sh --resume           # skip finished problems
#
set -euo pipefail

# ─── Defaults ─────────────────────────────────────────────────────────────────

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8001
P1_ITER=100
P2_ITER=200
RESULTS_DIR="results/alphaevolve_qwen7b_$(date +%Y%m%d_%H%M%S)"
SINGLE="" RESUME=false SERVE=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
API_BASE="http://localhost:${PORT}/v1"

# ─── Args ─────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve)       SERVE=true; shift ;;
    --problem)     SINGLE="$2"; shift 2 ;;
    --resume)      RESUME=true; shift ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --port)        PORT="$2"; API_BASE="http://localhost:${PORT}/v1"; shift 2 ;;
    --phase1-iter) P1_ITER="$2"; shift 2 ;;
    --phase2-iter) P2_ITER="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,8s/^# //p' "$0"; exit 0 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

# ─── Problem list: id|relative_path ──────────────────────────────────────────

PROBLEMS=(
  # "matmul|alphaevolve_math_problems/matmul"
  # "first_autocorr_ineq|alphaevolve_math_problems/first_autocorr_ineq"
  # "second_autocorr_ineq|alphaevolve_math_problems/second_autocorr_ineq"
  # "third_autocorr_ineq|alphaevolve_math_problems/third_autocorr_ineq"
  # "uncertainty_ineq|alphaevolve_math_problems/uncertainty_ineq"
  # "erdos_min_overlap|alphaevolve_math_problems/erdos_min_overlap"
  # "sums_diffs_finite_sets|alphaevolve_math_problems/sums_diffs_finite_sets"
  # "hexagon_packing_n11|alphaevolve_math_problems/hexagon_packing/11"
  # "hexagon_packing_n12|alphaevolve_math_problems/hexagon_packing/12"
  # "min_max_dist_d2|alphaevolve_math_problems/minimizing_max_min_dist/2"
  # "min_max_dist_d3|alphaevolve_math_problems/minimizing_max_min_dist/3"
  # "heilbronn_triangle|alphaevolve_math_problems/heilbronn_triangle"
  # "heilbronn_convex_n13|alphaevolve_math_problems/heilbronn_convex/13"
  # "heilbronn_convex_n14|alphaevolve_math_problems/heilbronn_convex/14"
  # "kissing_number|alphaevolve_math_problems/kissing_number"
  # "circle_packing|circle_packing"
  # "circle_packing_rect|alphaevolve_math_problems/circle_packing_rect"
)

# ─── vLLM server ─────────────────────────────────────────────────────────────

if $SERVE; then
  exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" \
    --gpu-memory-utilization 0.90 --max-model-len 30000 \
    --dtype auto --trust-remote-code
fi

# ─── Wait for vLLM ───────────────────────────────────────────────────────────

for i in $(seq 1 30); do
  curl -s "${API_BASE}/models" > /dev/null 2>&1 && break
  echo "Waiting for vLLM at ${API_BASE}... (${i}/30)"
  sleep 5
  [ "$i" -eq 30 ] && { echo "ERROR: vLLM not reachable. Start with: $0 --serve"; exit 1; }
done

# Detect model name as registered by vLLM
SERVED=$(curl -s "${API_BASE}/models" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null \
  || echo "$MODEL")

# ─── Resume support ──────────────────────────────────────────────────────────

if $RESUME; then
  LATEST=$(ls -d results/alphaevolve_qwen7b_* 2>/dev/null | sort | tail -1 || true)
  [ -n "$LATEST" ] && RESULTS_DIR="$LATEST" && echo "Resuming: ${RESULTS_DIR}"
fi
mkdir -p "$RESULTS_DIR"

cat > "${RESULTS_DIR}/run_config.json" <<EOF
{"model":"${MODEL}","served_model":"${SERVED}","api_base":"${API_BASE}",
 "phase1_iter":${P1_ITER},"phase2_iter":${P2_ITER},"start":"$(date -Iseconds)",
 "host":"$(hostname)","gpu":"$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"}
EOF

echo ""
echo "=== AlphaEvolve Benchmark: ${SERVED} | ${P1_ITER}+${P2_ITER} iters | ${RESULTS_DIR} ==="

# ─── Run one problem (two-phase) ─────────────────────────────────────────────

run_problem() {
  local pid="$1" edir="$ROOT/examples/$2" odir="$RESULTS_DIR/$1"

  # Resume: skip if summary exists
  $RESUME && [ -f "${odir}/summary.json" ] && echo "[SKIP] $pid" && return 0

  # Validate
  for f in initial_program.py evaluator.py config.yaml; do
    [ -f "${edir}/$f" ] || { echo "[ERROR] $pid: missing $f"; return 1; }
  done

  mkdir -p "$odir"
  echo ""
  echo "── $pid ──"

  # Generate phase configs from the problem's existing config
  python3 "$SCRIPT_DIR/gen_phase_configs.py" \
    "$edir/config.yaml" "$odir" \
    --model "$SERVED" --api-base "$API_BASE" \
    --phase1-iter "$P1_ITER" --phase2-iter "$P2_ITER"

  # Install deps
  [ -f "${edir}/requirements.txt" ] && pip install -q -r "${edir}/requirements.txt" 2>/dev/null || true

  local t0 oe_out latest_ckpt
  t0=$(date +%s)
  oe_out="${odir}/openevolve_output"

  # Phase 1
  echo "[$pid] Phase 1 (${P1_ITER} iters)..."
  python "$ROOT/openevolve-run.py" \
    "${edir}/initial_program.py" "${edir}/evaluator.py" \
    --config "${odir}/config_phase_1.yaml" --output "$oe_out" \
    2>&1 | tee "${odir}/evolution.log"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { _write_summary "$pid" "$odir" "$t0" "phase1_failed"; return 1; }

  # Find checkpoint
  latest_ckpt=$(ls -d "${oe_out}"/checkpoints/checkpoint_* 2>/dev/null | sort -V | tail -1)
  [ -z "$latest_ckpt" ] && { _write_summary "$pid" "$odir" "$t0" "no_checkpoint"; return 1; }

  # Phase 2
  echo "[$pid] Phase 2 (${P2_ITER} iters)..."
  python "$ROOT/openevolve-run.py" \
    "${latest_ckpt}/best_program.py" "${edir}/evaluator.py" \
    --config "${odir}/config_phase_2.yaml" \
    --checkpoint "$latest_ckpt" --output "$oe_out" \
    2>&1 | tee -a "${odir}/evolution.log"

  local status="completed"
  [ "${PIPESTATUS[0]}" -ne 0 ] && status="phase2_failed"

  # Copy best program out
  local final_ckpt
  final_ckpt=$(ls -d "${oe_out}"/checkpoints/checkpoint_* 2>/dev/null | sort -V | tail -1)
  [ -n "$final_ckpt" ] && cp -f "${final_ckpt}"/best_program*.{py,json} "$odir/" 2>/dev/null || true

  _write_summary "$pid" "$odir" "$t0" "$status"
}

_write_summary() {
  local pid="$1" odir="$2" t0="$3" status="$4"
  local elapsed=$(( $(date +%s) - t0 ))
  python3 -c "
import json, os
info_path = os.path.join('$odir', 'best_program_info.json')
metrics = {}
if os.path.exists(info_path):
    with open(info_path) as f: metrics = json.load(f).get('metrics', {})
summary = {'problem':'$pid','status':'$status','elapsed_seconds':$elapsed,
           'elapsed_human':'${elapsed}s','best_metrics':metrics}
with open(os.path.join('$odir','summary.json'),'w') as f:
    json.dump(summary, f, indent=2, default=str)
"
  echo "[$pid] $status (${elapsed}s)"
}

# ─── Main ─────────────────────────────────────────────────────────────────────

cd "$ROOT"
PASSED=0 FAILED=0

if [ -n "$SINGLE" ]; then
  # Single problem mode
  for entry in "${PROBLEMS[@]}"; do
    IFS='|' read -r pid ppath <<< "$entry"
    [ "$pid" = "$SINGLE" ] && { run_problem "$pid" "$ppath" && PASSED=1 || FAILED=1; break; }
  done
  [ $((PASSED+FAILED)) -eq 0 ] && {
    echo "Unknown problem: $SINGLE. Available:"
    printf '  %s\n' "${PROBLEMS[@]%%|*}"
    exit 1
  }
else
  for i in "${!PROBLEMS[@]}"; do
    IFS='|' read -r pid ppath <<< "${PROBLEMS[$i]}"
    echo ""
    echo "═══ [$((i+1))/${#PROBLEMS[@]}] $pid ═══"
    run_problem "$pid" "$ppath" && PASSED=$((PASSED+1)) || FAILED=$((FAILED+1))
  done
fi

echo ""
echo "=== Done: ${PASSED} passed, ${FAILED} failed ==="
echo ""

# Final report via analyze.py
python3 "$SCRIPT_DIR/analyze.py" --benchmark "$RESULTS_DIR"
