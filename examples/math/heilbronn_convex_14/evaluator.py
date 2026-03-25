"""Evaluator for Heilbronn convex (14 points)."""
import subprocess, sys, json

BENCHMARK = 0.027835571458482138

def evaluate(program_path):
    script = f"""
import json, sys, os, time
import numpy as np
import itertools
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
if points.shape != (14, 2):
    print(json.dumps({{"error": f"bad shape {{points.shape}}"}})); sys.exit(0)
from scipy.spatial import ConvexHull
hull = ConvexHull(points)
hull_area = hull.volume
def tri_area(a,b,c):
    return abs(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))/2
min_a = min(tri_area(p1,p2,p3) for p1,p2,p3 in itertools.combinations(points, 3))
norm = min_a / hull_area if hull_area > 0 else 0
score = norm / {BENCHMARK}
print(json.dumps({{"combined_score": score, "min_area_normalized": norm, "eval_time": eval_time}}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
        for line in reversed(r.stdout.strip().splitlines()):
            if line.strip().startswith("{"): return json.loads(line.strip())
        return {{"combined_score": 0.0, "error": r.stderr[-500:]}}
    except Exception as e:
        return {{"combined_score": 0.0, "error": str(e)}}
