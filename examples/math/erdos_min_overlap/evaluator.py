"""Evaluator for Erdos minimum overlap — minimize C5 upper bound."""
import subprocess, sys, json

BENCHMARK = 0.38092303510845016

def evaluate(program_path):
    script = f"""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(r'{program_path}'))
module_name = os.path.splitext(os.path.basename(r'{program_path}'))[0]
start = time.time()
try:
    program = __import__(module_name)
    h_values, c5_bound, n_points = program.run()
except Exception as e:
    print(json.dumps({{"error": str(e)}})); sys.exit(0)
eval_time = time.time() - start
h = np.array(h_values, dtype=float)
if h.shape != (n_points,):
    print(json.dumps({{"error": f"bad shape {{h.shape}}"}})); sys.exit(0)
if np.any(h < 0) or np.any(h > 1):
    print(json.dumps({{"error": f"h not in [0,1]"}})); sys.exit(0)
dx = 2.0 / n_points
integral_h = np.sum(h) * dx
if not np.isclose(integral_h, 1.0, atol=1e-3):
    print(json.dumps({{"error": f"integral={{integral_h}}"}})); sys.exit(0)
j = 1.0 - h
corr = np.correlate(h, j, mode="full") * dx
computed = float(np.max(corr))
if not np.isclose(computed, float(c5_bound), atol=1e-4):
    print(json.dumps({{"error": f"C5 mismatch: {{c5_bound}} vs {{computed}}"}})); sys.exit(0)
score = {BENCHMARK} / float(c5_bound)
print(json.dumps({{"combined_score": score, "c5_bound": float(c5_bound), "eval_time": eval_time}}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
        for line in reversed(r.stdout.strip().splitlines()):
            if line.strip().startswith("{"): return json.loads(line.strip())
        return {{"combined_score": 0.0, "error": r.stderr[-500:]}}
    except Exception as e:
        return {{"combined_score": 0.0, "error": str(e)}}
