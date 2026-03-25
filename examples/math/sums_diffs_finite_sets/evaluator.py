"""Evaluator for sums and differences of finite sets — maximize C6."""
import subprocess, sys, json

BENCHMARK = 1.158417281556896

def evaluate(program_path):
    script = f"""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(r'{program_path}'))
module_name = os.path.splitext(os.path.basename(r'{program_path}'))[0]
start = time.time()
try:
    program = __import__(module_name)
    u_set, c6_bound = program.run()
except Exception as e:
    print(json.dumps({{"error": str(e)}})); sys.exit(0)
eval_time = time.time() - start
U = np.array(u_set, dtype=int)
if U.ndim != 1:
    print(json.dumps({{"error": "U must be 1D"}})); sys.exit(0)
if 0 not in U:
    print(json.dumps({{"error": "U must contain 0"}})); sys.exit(0)
if np.any(U < 0):
    print(json.dumps({{"error": "negative elements"}})); sys.exit(0)
max_u = int(np.max(U))
if max_u == 0:
    print(json.dumps({{"error": "trivial set"}})); sys.exit(0)
u_plus = np.unique(U[:, None] + U[None, :])
u_minus = np.unique(U[:, None] - U[None, :])
ratio = len(u_minus) / len(u_plus)
c6 = 1 + np.log(ratio) / np.log(2 * max_u + 1)
if abs(c6 - float(c6_bound)) > 0.01:
    print(json.dumps({{"error": f"C6 mismatch: {{c6_bound}} vs {{c6}}"}})); sys.exit(0)
score = float(c6) / {BENCHMARK}
print(json.dumps({{"combined_score": score, "c6": float(c6), "eval_time": eval_time}}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
        for line in reversed(r.stdout.strip().splitlines()):
            if line.strip().startswith("{"): return json.loads(line.strip())
        return {{"combined_score": 0.0, "error": r.stderr[-500:]}}
    except Exception as e:
        return {{"combined_score": 0.0, "error": str(e)}}
