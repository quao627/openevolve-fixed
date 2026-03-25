"""Evaluator for second autocorrelation inequality — maximize C2."""
import subprocess, sys, json

BENCHMARK = 0.8962799441554086

def evaluate(program_path):
    script = f"""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(r'{program_path}'))
module_name = os.path.splitext(os.path.basename(r'{program_path}'))[0]
start = time.time()
try:
    program = __import__(module_name)
    f_values, c2_achieved, loss, n_points = program.run()
except Exception as e:
    print(json.dumps({{"error": str(e)}})); sys.exit(0)
eval_time = time.time() - start
f = np.array(f_values, dtype=float)
if f.shape != (n_points,):
    print(json.dumps({{"error": f"bad shape {{f.shape}}"}})); sys.exit(0)
if np.any(f < -1e-6):
    print(json.dumps({{"error": "f has negative values"}})); sys.exit(0)
conv = np.convolve(f, f, mode="full")
n_conv = len(conv)
padded = np.concatenate([[0], conv, [0]])
l2_sq = 0.0
for i in range(n_conv + 1):
    a, b = padded[i], padded[i+1]
    l2_sq += (a*a + a*b + b*b) / 3.0
l2_sq /= (n_conv + 1)
norm_1 = np.sum(np.abs(conv)) / (n_conv + 1)
norm_inf = np.max(np.abs(conv))
c2_computed = l2_sq / (norm_1 * norm_inf) if norm_1 * norm_inf > 0 else 0
score = c2_computed / {BENCHMARK}
print(json.dumps({{"combined_score": score, "c2": float(c2_computed), "eval_time": eval_time}}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
        for line in reversed(r.stdout.strip().splitlines()):
            if line.strip().startswith("{"): return json.loads(line.strip())
        return {{"combined_score": 0.0, "error": r.stderr[-500:]}}
    except Exception as e:
        return {{"combined_score": 0.0, "error": str(e)}}
