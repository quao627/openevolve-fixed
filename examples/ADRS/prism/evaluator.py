"""Evaluator for Prism — LLM model placement on GPU cluster."""
import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
from dataclasses import dataclass

GPU_MEM_SIZE = 80

@dataclass
class Model:
    model_name: str
    model_size: int
    req_rate: int
    slo: int
    cur_gpu_id: int

def safe_float(value):
    try:
        if np.isnan(value) or np.isinf(value): return 0.0
        return float(value)
    except: return 0.0

def calculate_kvcache_pressure(placement):
    max_kvpr = float('-inf')
    for gpu_id, models in placement.items():
        total_size = sum(m.model_size for m in models)
        total_rate = sum(m.req_rate / m.slo for m in models)
        kvpr = total_rate / (GPU_MEM_SIZE - total_size) if GPU_MEM_SIZE - total_size > 0 else 1000000
        max_kvpr = max(max_kvpr, kvpr)
    return max_kvpr

def generate_test_cases(num_tests=50):
    np.random.seed(42)
    cases = []
    for _ in range(num_tests):
        gpu_num = np.random.randint(5, 10)
        models = [Model(f"model_{j}", np.random.randint(10,30), np.random.randint(1,10),
                       np.random.randint(5,10), j) for j in range(gpu_num*2)]
        cases.append((gpu_num, models))
    return cases

def evaluate(program_path):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        if not hasattr(program, "compute_model_placement"):
            return {"combined_score": 0.0, "error": "Missing compute_model_placement"}

        cases = generate_test_cases()
        all_kvpr = []
        ok = 0
        for gpu_num, models in cases:
            try:
                with concurrent.futures.ThreadPoolExecutor(1) as ex:
                    future = ex.submit(program.compute_model_placement, gpu_num, models)
                    result = future.result(timeout=10)
                if isinstance(result, dict):
                    kvpr = calculate_kvcache_pressure(result)
                    all_kvpr.append(safe_float(kvpr))
                    ok += 1
            except: pass

        if ok == 0:
            return {"combined_score": 0.0, "error": "All test cases failed"}
        avg_kvpr = np.mean(all_kvpr)
        inv_kvpr = 1.0 / avg_kvpr if avg_kvpr > 0 else 0
        success_rate = ok / len(cases)
        return {"combined_score": safe_float(inv_kvpr + success_rate),
                "inv_kvpr": safe_float(inv_kvpr), "success_rate": safe_float(success_rate)}
    except Exception as e:
        traceback.print_exc()
        return {"combined_score": 0.0, "error": str(e)}
