"""Evaluator for circle packing in rectangle (n=21, perimeter=4)."""
import subprocess, sys, json

TARGET = 2.3658321334167627

def evaluate(program_path):
    script = f"""
import json, sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(r'{program_path}'))
module_name = os.path.splitext(os.path.basename(r'{program_path}'))[0]
start = time.time()
try:
    program = __import__(module_name)
    circles = program.run()
except Exception as e:
    print(json.dumps({{"error": str(e)}})); sys.exit(0)
eval_time = time.time() - start
circles = np.array(circles, dtype=float)
if circles.shape != (21, 3):
    print(json.dumps({{"combined_score": 0.0, "error": f"bad shape {{circles.shape}}"}})); sys.exit(0)
xs, ys, rs = circles[:,0], circles[:,1], circles[:,2]
if np.any(rs < 0) or np.isnan(circles).any():
    print(json.dumps({{"combined_score": 0.0, "error": "negative radii or NaN"}})); sys.exit(0)
TOL = 1e-6
for i in range(21):
    for j in range(i+1, 21):
        d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
        if d < rs[i]+rs[j]-TOL:
            print(json.dumps({{"combined_score": 0.0, "error": f"overlap {{i}} {{j}}"}})); sys.exit(0)
x_min = np.min(xs - rs); x_max = np.max(xs + rs)
y_min = np.min(ys - rs); y_max = np.max(ys + rs)
w = x_max - x_min; h = y_max - y_min
peri = 2*(w+h)
if peri > 4 + TOL:
    print(json.dumps({{"combined_score": 0.0, "error": f"perimeter {{peri}} > 4"}})); sys.exit(0)
s = float(np.sum(rs))
print(json.dumps({{"combined_score": s/{TARGET}, "sum_radii": s, "eval_time": eval_time}}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
        for line in reversed(r.stdout.strip().splitlines()):
            if line.strip().startswith("{"): return json.loads(line.strip())
        return {{"combined_score": 0.0, "error": r.stderr[-500:]}}
    except Exception as e:
        return {{"combined_score": 0.0, "error": str(e)}}
