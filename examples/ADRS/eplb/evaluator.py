"""Evaluator for EPLB (Expert Parallelism Load Balancer).

Requires expert-load.json in the same directory.
Download: wget https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json
"""
import functools
import importlib.util
import json
import time
import traceback
import os

import torch

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKLOAD_PATH = os.path.join(_CURRENT_DIR, "expert-load.json")
REBALANCE_INTERVAL = 100
NUM_REPLICAS = 288
NUM_GROUPS = 8
NUM_GPUS = 32
NUM_NODES = 4

@functools.cache
def load_workloads(path):
    with open(path, "r") as f:
        data = json.load(f)
    total_len = len(data['load_history'])
    workloads = []
    for i in range(0, total_len, REBALANCE_INTERVAL):
        load = torch.tensor([x['logical_expert_load'] for x in data['load_history'][i:min(i+REBALANCE_INTERVAL, total_len)]]).sum(dim=0)
        workloads.append(load)
    return workloads

def simulate_inference(log2phy, logcnt, workload):
    num_layers, num_logical_experts = workload.shape
    total_physical_load = torch.zeros(num_layers, NUM_REPLICAS, dtype=torch.float)
    for layer_id in range(num_layers):
        for logical_id in range(num_logical_experts):
            logical_load = workload[layer_id][logical_id].item()
            if logical_load <= 0: continue
            num_reps = int(logcnt[layer_id][logical_id].item())
            if num_reps <= 0: continue
            physical_ids = log2phy[layer_id][logical_id][:num_reps]
            total_physical_load[layer_id, physical_ids] += logical_load / num_reps
    total_load = total_physical_load.sum()
    if total_load == 0: return 0.0, 0.0
    expert_avg = total_physical_load.mean(dim=1).sum().item()
    expert_max = total_physical_load.max(dim=1).values.sum().item()
    bal_expert = expert_avg / expert_max
    gpu_load = total_physical_load.view(num_layers, NUM_GPUS, -1).sum(dim=2)
    avg_load = gpu_load.mean(dim=1).sum().item()
    max_load = gpu_load.max(dim=1).values.sum().item()
    bal_gpu = avg_load / max_load if max_load > 0 else 0.0
    return bal_gpu, bal_expert

def evaluate(program_path):
    try:
        workloads = load_workloads(WORKLOAD_PATH)
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        if not hasattr(program, "rebalance_experts"):
            return {"combined_score": 0.0, "error": "Missing rebalance_experts"}

        bal_experts, speeds = [], []
        for i in range(len(workloads) - 1):
            t0 = time.perf_counter()
            _, log2phy, logcnt = program.rebalance_experts(
                workloads[i], NUM_REPLICAS, NUM_GROUPS, NUM_NODES, NUM_GPUS)
            t1 = time.perf_counter()
            _, bal_e = simulate_inference(log2phy, logcnt, workloads[i+1])
            t2 = time.perf_counter()
            bal_experts.append(bal_e)
            speeds.append(t2 - t0)

        avg_bal = sum(bal_experts) / len(bal_experts)
        avg_time = sum(speeds) / len(speeds)
        speed_score = 0.002 / avg_time
        combined = (avg_bal + speed_score) / 2
        return {"combined_score": float(combined), "balancedness": float(avg_bal),
                "speed_score": float(speed_score), "avg_time": float(avg_time)}
    except Exception as e:
        traceback.print_exc()
        return {"combined_score": 0.0, "error": str(e)}
