"""Evaluator for matrix multiplication tensor decomposition (n=2, m=4, p=5)."""
import subprocess, sys, json

BENCHMARK = 32  # benchmark rank

def evaluate(program_path):
    script = f"""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(r'{program_path}'))
module_name = os.path.splitext(os.path.basename(r'{program_path}'))[0]
start = time.time()
try:
    program = __import__(module_name)
    decomposition, n, m, p, loss, rank = program.run()
except Exception as e:
    print(json.dumps({{"error": str(e)}})); sys.exit(0)
eval_time = time.time() - start
U, V, W = decomposition
U, V, W = np.array(U), np.array(V), np.array(W)
if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (n*p, rank):
    print(json.dumps({{"error": f"bad shapes U={{U.shape}} V={{V.shape}} W={{W.shape}}"}})); sys.exit(0)
T = np.zeros((n*m, m*p, n*p))
for i in range(n):
    for j in range(m):
        for k in range(p):
            T[i*m+j, j*p+k, k*n+i] = 1
R = np.einsum("ir,jr,kr->ijk", U, V, W)
if not np.array_equal(T, R):
    print(json.dumps({{"error": "decomposition does not match tensor"}})); sys.exit(0)
score = {BENCHMARK} / rank
print(json.dumps({{"combined_score": score, "rank": rank, "eval_time": eval_time}}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
        for line in reversed(r.stdout.strip().splitlines()):
            if line.strip().startswith("{"): return json.loads(line.strip())
        return {{"combined_score": 0.0, "error": r.stderr[-500:]}}
    except Exception as e:
        return {{"combined_score": 0.0, "error": str(e)}}
