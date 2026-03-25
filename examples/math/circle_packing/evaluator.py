"""Evaluator for circle packing (n=26) — maximize sum of radii in unit square."""
import importlib.util, subprocess, sys, json, time, traceback, os

TARGET = 2.635977  # AlphaEvolve benchmark

def evaluate(program_path):
    script = f"""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(r'{program_path}'))
module_name = os.path.splitext(os.path.basename(r'{program_path}'))[0]
start = time.time()
try:
    program = __import__(module_name)
    centers, radii, sum_radii = program.run()
except Exception as e:
    print(json.dumps({{"error": str(e)}})); sys.exit(0)
eval_time = time.time() - start
centers = np.array(centers, dtype=float); radii = np.array(radii, dtype=float)
if np.isnan(centers).any() or np.isnan(radii).any():
    print(json.dumps({{"combined_score": 0.0, "error": "NaN"}})); sys.exit(0)
if centers.shape != (26, 2) or radii.shape != (26,):
    print(json.dumps({{"combined_score": 0.0, "error": f"bad shape {{centers.shape}} {{radii.shape}}"}})); sys.exit(0)
if np.any(radii < 0):
    print(json.dumps({{"combined_score": 0.0, "error": "negative radii"}})); sys.exit(0)
for i in range(26):
    x, y, r = centers[i][0], centers[i][1], radii[i]
    if x-r < -1e-6 or y-r < -1e-6 or x+r > 1+1e-6 or y+r > 1+1e-6:
        print(json.dumps({{"combined_score": 0.0, "error": f"circle {{i}} outside"}})); sys.exit(0)
for i in range(26):
    for j in range(i+1, 26):
        d = np.sqrt(np.sum((centers[i]-centers[j])**2))
        if d < radii[i]+radii[j]-1e-6:
            print(json.dumps({{"combined_score": 0.0, "error": f"overlap {{i}} {{j}}"}})); sys.exit(0)
s = float(np.sum(radii))
print(json.dumps({{"combined_score": s/{TARGET}, "sum_radii": s, "eval_time": eval_time}}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
        for line in reversed(r.stdout.strip().splitlines()):
            if line.strip().startswith("{"):
                return json.loads(line.strip())
        return {{"combined_score": 0.0, "error": r.stderr[-500:]}}
    except Exception as e:
        return {{"combined_score": 0.0, "error": str(e)}}
