"""Evaluator for hexagon packing (n=12) — minimize outer hexagon side length."""
import subprocess, sys, json

BENCHMARK_SIDE = 3.9419123
BENCHMARK = 1.0 / BENCHMARK_SIDE

def evaluate(program_path):
    script = f"""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(r'{program_path}'))
module_name = os.path.splitext(os.path.basename(r'{program_path}'))[0]
start = time.time()
try:
    program = __import__(module_name)
    inner_hex_data, outer_hex_data, outer_side = program.run()
except Exception as e:
    print(json.dumps({{"error": str(e)}})); sys.exit(0)
eval_time = time.time() - start
inner = np.array(inner_hex_data, dtype=float)
outer = np.array(outer_hex_data, dtype=float)
if inner.shape != (12, 3):
    print(json.dumps({{"error": f"bad inner shape {{inner.shape}}"}})); sys.exit(0)
if outer.shape != (3,):
    print(json.dumps({{"error": f"bad outer shape {{outer.shape}}"}})); sys.exit(0)
if np.isnan(inner).any() or np.isnan(outer).any() or np.isnan(outer_side):
    print(json.dumps({{"combined_score": 0.0, "error": "NaN"}})); sys.exit(0)
inv = 1.0 / float(outer_side) if float(outer_side) > 0 else 0.0
score = inv / {BENCHMARK}
print(json.dumps({{"combined_score": score, "outer_side": float(outer_side), "eval_time": eval_time}}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
        for line in reversed(r.stdout.strip().splitlines()):
            if line.strip().startswith("{"): return json.loads(line.strip())
        return {{"combined_score": 0.0, "error": r.stderr[-500:]}}
    except Exception as e:
        return {{"combined_score": 0.0, "error": str(e)}}
