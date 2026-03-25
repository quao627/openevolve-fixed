"""Evaluator for third autocorrelation inequality — minimize C3."""
import subprocess, sys, json

BENCHMARK = 1.4556427953745406

def evaluate(program_path):
    script = f"""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(r'{program_path}'))
module_name = os.path.splitext(os.path.basename(r'{program_path}'))[0]
start = time.time()
try:
    program = __import__(module_name)
    f_values, c3_achieved, loss, n_points = program.run()
except Exception as e:
    print(json.dumps({{"error": str(e)}})); sys.exit(0)
eval_time = time.time() - start
f = np.array(f_values, dtype=float)
if f.shape != (n_points,):
    print(json.dumps({{"error": f"bad shape {{f.shape}}"}})); sys.exit(0)
dx = 0.5 / n_points
conv = np.convolve(f, f, mode="full") * dx
max_abs_conv = float(np.max(np.abs(conv)))
integral_f_sq = (np.sum(f) * dx) ** 2
if integral_f_sq < 1e-12:
    print(json.dumps({{"error": "integral near zero"}})); sys.exit(0)
c3_computed = max_abs_conv / integral_f_sq
if abs(c3_computed - float(c3_achieved)) > 1e-3:
    print(json.dumps({{"error": f"C3 mismatch: {{c3_achieved}} vs {{c3_computed}}"}})); sys.exit(0)
score = {BENCHMARK} / c3_computed
print(json.dumps({{"combined_score": score, "c3": c3_computed, "eval_time": eval_time}}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
        for line in reversed(r.stdout.strip().splitlines()):
            if line.strip().startswith("{"): return json.loads(line.strip())
        return {{"combined_score": 0.0, "error": r.stderr[-500:]}}
    except Exception as e:
        return {{"combined_score": 0.0, "error": str(e)}}
