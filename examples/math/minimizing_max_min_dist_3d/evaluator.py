"""Evaluator for minimizing max-min distance (3D, 14 points)."""
import subprocess, sys, json

BENCHMARK = 1.0 / 4.165849767

def evaluate(program_path):
    script = f"""
import json, sys, os, time
import numpy as np
from scipy.spatial.distance import pdist
sys.path.insert(0, os.path.dirname(r'{program_path}'))
module_name = os.path.splitext(os.path.basename(r'{program_path}'))[0]
start = time.time()
try:
    program = __import__(module_name)
    points = program.run()
except Exception as e:
    print(json.dumps({{"error": str(e)}})); sys.exit(0)
eval_time = time.time() - start
points = np.array(points, dtype=float)
if points.shape != (14, 3):
    print(json.dumps({{"error": f"bad shape {{points.shape}}"}})); sys.exit(0)
if np.isnan(points).any():
    print(json.dumps({{"error": "NaN"}})); sys.exit(0)
dists = pdist(points)
min_d, max_d = np.min(dists), np.max(dists)
if max_d == 0:
    print(json.dumps({{"error": "max distance is 0"}})); sys.exit(0)
ratio_sq = (min_d / max_d) ** 2
score = ratio_sq / {BENCHMARK}
print(json.dumps({{"combined_score": score, "ratio_squared": ratio_sq, "eval_time": eval_time}}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
        for line in reversed(r.stdout.strip().splitlines()):
            if line.strip().startswith("{"): return json.loads(line.strip())
        return {{"combined_score": 0.0, "error": r.stderr[-500:]}}
    except Exception as e:
        return {{"combined_score": 0.0, "error": str(e)}}
