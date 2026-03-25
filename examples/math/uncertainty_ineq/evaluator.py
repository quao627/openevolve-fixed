"""Evaluator for uncertainty inequality — minimize C4 via Hermite polynomial."""
import subprocess, sys, json

BENCHMARK = 0.3215872333529007

def evaluate(program_path):
    script = f"""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(r'{program_path}'))
module_name = os.path.splitext(os.path.basename(r'{program_path}'))[0]
start = time.time()
try:
    program = __import__(module_name)
    coeffs, c4_bound, r_max = program.run()
except Exception as e:
    print(json.dumps({{"error": str(e)}})); sys.exit(0)
eval_time = time.time() - start
coeffs = np.array(coeffs, dtype=float)
if np.isnan(coeffs).any():
    print(json.dumps({{"error": "NaN in coefficients"}})); sys.exit(0)
c4_check = float(r_max)**2 / (2*np.pi)
if abs(c4_check - float(c4_bound)) / max(abs(c4_check), 1e-12) > 1e-6:
    print(json.dumps({{"error": f"C4 mismatch: {{c4_bound}} vs {{c4_check}}"}})); sys.exit(0)
score = {BENCHMARK} / float(c4_bound)
print(json.dumps({{"combined_score": score, "c4": float(c4_bound), "eval_time": eval_time}}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
        for line in reversed(r.stdout.strip().splitlines()):
            if line.strip().startswith("{"): return json.loads(line.strip())
        return {{"combined_score": 0.0, "error": r.stderr[-500:]}}
    except Exception as e:
        return {{"combined_score": 0.0, "error": str(e)}}
