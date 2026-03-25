"""Evaluator for Heilbronn triangle (11 points in equilateral triangle)."""
import subprocess, sys, json

BENCHMARK = 0.036529889880030156

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
if points.shape != (11, 2):
    print(json.dumps({{"error": f"bad shape {{points.shape}}"}})); sys.exit(0)
TOL = 1e-6
for i, (x, y) in enumerate(points):
    if not (y >= -TOL and np.sqrt(3)*x <= np.sqrt(3)-y+TOL and y <= np.sqrt(3)*x+TOL):
        print(json.dumps({{"error": f"point {{i}} outside triangle"}})); sys.exit(0)
def tri_area(a,b,c):
    return abs(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))/2
eq_area = tri_area(np.array([0,0]), np.array([1,0]), np.array([0.5, np.sqrt(3)/2]))
min_a = min(tri_area(p1,p2,p3) for p1,p2,p3 in itertools.combinations(points, 3))
norm = min_a / eq_area
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
