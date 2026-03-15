"""
Evaluator for the Performance Take-Home task.

Runs the evolved KernelBuilder on the VLIW SIMD simulator, checks correctness
across multiple random seeds, and returns a score based on cycle count.
"""

import sys
import os
import random
import time

PROBLEM_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "problem_src")
sys.path.insert(0, PROBLEM_SRC)

from problem import (
    Machine,
    Tree,
    Input,
    build_mem_image,
    reference_kernel2,
    N_CORES,
)

BASELINE = 147734
# Best known human result
BEST_KNOWN = 1363


def run_kernel(kb, forest_height, rounds, batch_size, seed=None):
    """Run the kernel and return (cycles, correct)."""
    if seed is not None:
        random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
    )
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    # Check correctness against reference
    ref_mem = None
    for ref_mem in reference_kernel2(mem):
        pass

    inp_values_p = ref_mem[6]
    correct = (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    )

    return machine.cycle, correct


def evaluate(program_path: str):
    try:
        abs_program_path = os.path.abspath(program_path)
        program_dir = os.path.dirname(abs_program_path)
        module_name = os.path.splitext(os.path.basename(program_path))[0]

        try:
            sys.path.insert(0, program_dir)
            # Remove cached module if any
            if module_name in sys.modules:
                del sys.modules[module_name]
            program = __import__(module_name)
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        KernelBuilder = program.KernelBuilder

        # Build the kernel once (it's deterministic)
        start_time = time.time()
        kb = KernelBuilder()
        kb.build_kernel(
            forest_height=10,
            n_nodes=2**10 - 1,  # 1023 nodes in a tree of height 10
            batch_size=256,
            rounds=16,
        )
        build_time = time.time() - start_time

        # Check correctness across multiple random seeds
        correctness_seeds = [42, 123, 456, 789]
        for seed in correctness_seeds:
            cycles, correct = run_kernel(kb, 10, 16, 256, seed=seed)
            if not correct:
                return {
                    "combined_score": 0.0,
                    "error": f"Incorrect output with seed={seed}",
                }

        # Measure performance (use a fixed seed for consistency)
        random.seed(123)
        cycles, correct = run_kernel(kb, 10, 16, 256, seed=123)

        if not correct:
            return {"combined_score": 0.0, "error": "Incorrect output"}

        # Score: higher is better
        # speedup over baseline, normalized so that best_known = 1.0
        speedup = BASELINE / cycles
        normalized_score = speedup / (BASELINE / BEST_KNOWN)  # 1.0 at best known

        # Also provide a simpler combined_score = baseline/cycles for ranking
        combined_score = speedup

        eval_time = time.time() - start_time

        return {
            "combined_score": float(combined_score),
            "cycles": int(cycles),
            "speedup": float(speedup),
            "normalized_score": float(normalized_score),
            "build_time": float(build_time),
            "eval_time": float(eval_time),
            "num_instructions": len(kb.instrs),
        }
    except Exception as e:
        import traceback
        return {"combined_score": 0.0, "error": f"{e}\n{traceback.format_exc()}"}
