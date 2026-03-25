"""Evaluator for transaction scheduling — minimize makespan.

Requires txn_simulator.py and workloads.py in the same directory.
"""
import importlib.util
import os
import sys
import subprocess
import json
import time
import traceback
import pickle
import tempfile


def evaluate(program_path):
    sched_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        script = f"""
import sys, os, pickle, traceback
sys.path.insert(0, os.path.dirname(r'{program_path}'))
sys.path.insert(0, r'{sched_dir}')
try:
    spec = __import__('importlib.util').util.spec_from_file_location("program", r'{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    makespan, schedule = program.get_random_costs()
    with open(r'{tmp.name}.results', 'wb') as f:
        pickle.dump({{'makespan': makespan, 'schedule': schedule}}, f)
except Exception as e:
    traceback.print_exc()
    with open(r'{tmp.name}.results', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
"""
        tmp.write(script.encode())
        tmp_path = tmp.name

    results_path = f"{tmp_path}.results"
    try:
        proc = subprocess.run([sys.executable, tmp_path],
                              capture_output=True, text=True, timeout=600)
        if not os.path.exists(results_path):
            return {"combined_score": 0.0, "error": proc.stderr[-500:]}
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        if "error" in results:
            return {"combined_score": 0.0, "error": results["error"]}
        makespan = results["makespan"]
        schedule = results["schedule"]
        # Validate
        valid = all(sorted(s) == list(range(len(s))) for s in schedule)
        combined = 1000000 / (1 + makespan) if valid else 0.0
        return {"combined_score": float(combined), "makespan": float(makespan),
                "valid": valid, "num_schedules": len(schedule)}
    except subprocess.TimeoutExpired:
        return {"combined_score": 0.0, "error": "timeout"}
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}
    finally:
        for p in [tmp_path, results_path]:
            if os.path.exists(p):
                os.unlink(p)
