"""
Evaluator for the TriMul kernel engineering task.

Runs the submission through the eval harness (correctness + benchmarking)
and returns a combined_score = 1000 / geometric_mean(runtime_us).
"""

import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Eval harness files live alongside this evaluator
EVAL_DIR = Path(__file__).parent / "eval"

# Test and benchmark specs from task.yml
TESTS = [
    "seqlen: 32; bs: 1; dim: 128; hiddendim: 128; seed: 9371; nomask: True; distribution: normal",
    "seqlen: 32; bs: 1; dim: 128; hiddendim: 128; seed: 1092; nomask: False; distribution: normal",
    "seqlen: 64; bs: 2; dim: 256; hiddendim: 128; seed: 2291; nomask: True; distribution: normal",
    "seqlen: 64; bs: 2; dim: 256; hiddendim: 128; seed: 210284; nomask: False; distribution: normal",
    "seqlen: 128; bs: 1; dim: 768; hiddendim: 128; seed: 81934; nomask: True; distribution: normal",
    "seqlen: 256; bs: 1; dim: 128; hiddendim: 128; seed: 1932; nomask: True; distribution: normal",
    "seqlen: 256; bs: 1; dim: 128; hiddendim: 128; seed: 10432; nomask: False; distribution: normal",
    "seqlen: 768; bs: 2; dim: 128; hiddendim: 128; seed: 731; nomask: True; distribution: normal",
]

BENCHMARKS = [
    "seqlen: 256; bs: 2; dim: 128; hiddendim: 128; seed: 9371; nomask: True; distribution: normal",
    "seqlen: 768; bs: 1; dim: 128; hiddendim: 128; seed: 381; nomask: True; distribution: cauchy",
    "seqlen: 256; bs: 2; dim: 384; hiddendim: 128; seed: 2301; nomask: False; distribution: normal",
    "seqlen: 512; bs: 1; dim: 128; hiddendim: 128; seed: 12819; nomask: True; distribution: normal",
    "seqlen: 1024; bs: 1; dim: 128; hiddendim: 128; seed: 381; nomask: True; distribution: cauchy",
    "seqlen: 768; bs: 1; dim: 384; hiddendim: 128; seed: 481; nomask: False; distribution: normal",
    "seqlen: 1024; bs: 1; dim: 384; hiddendim: 128; seed: 23291; nomask: True; distribution: normal",
]


def _parse_popcorn_output(output: str) -> dict[str, str]:
    results = {}
    for line in output.strip().splitlines():
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            results[key.strip()] = value.strip()
    return results


def _compute_geomean(values: list[float]) -> float:
    if not values:
        return 0.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


def _run_eval(submission_path: str, mode: str, specs: list[str], timeout: int = 600) -> dict[str, str]:
    """Run the eval harness in a temp directory with the given submission."""
    with tempfile.TemporaryDirectory(prefix="oe_kernel_") as tmpdir:
        # Copy eval harness files
        for f in EVAL_DIR.iterdir():
            if f.name not in ("grader.py",) and f.is_file():
                shutil.copy2(f, tmpdir)

        # Copy submission
        shutil.copy2(submission_path, os.path.join(tmpdir, "submission.py"))

        # Write test spec file
        spec_path = os.path.join(tmpdir, f"{mode}.txt")
        with open(spec_path, "w") as f:
            for spec in specs:
                f.write(spec + "\n")

        # Run eval.py with POPCORN_FD protocol
        read_fd, write_fd = os.pipe()
        env = os.environ.copy()
        env["POPCORN_FD"] = str(write_fd)

        cmd = ["/usr/bin/python3", "eval.py", mode, f"{mode}.txt"]

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                pass_fds=(write_fd,),
            )
            os.close(write_fd)
            write_fd = -1

            popcorn_output = []

            def _read_popcorn():
                with os.fdopen(read_fd, "r") as f:
                    popcorn_output.append(f.read())

            reader = threading.Thread(target=_read_popcorn, daemon=True)
            reader.start()

            stdout, stderr = proc.communicate(timeout=timeout)
            reader.join(timeout=10)

            if stderr:
                err_text = stderr.decode("utf-8", errors="replace")[:2000]
                print(f"[eval stderr] {err_text}", file=sys.stderr)

            return _parse_popcorn_output(popcorn_output[0] if popcorn_output else "")

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return {"check": "fail", "error": f"Eval timed out after {timeout}s"}
        except Exception as e:
            return {"check": "fail", "error": str(e)}
        finally:
            if write_fd >= 0:
                try:
                    os.close(write_fd)
                except OSError:
                    pass


def evaluate(program_path: str) -> dict:
    """
    Evaluate a kernel engineering submission.

    Returns:
        dict with combined_score (1000 / geomean_runtime_us) and metadata.
    """
    start_time = time.time()

    # Verify submission has custom_kernel
    try:
        code = Path(program_path).read_text()
        if "custom_kernel" not in code:
            return {"combined_score": 0.0, "error": "No custom_kernel function found"}
    except Exception as e:
        return {"combined_score": 0.0, "error": f"Cannot read submission: {e}"}

    # Step 1: Correctness tests (smaller subset for speed)
    print("[evaluator] Running correctness tests...")
    test_results = _run_eval(program_path, mode="test", specs=TESTS, timeout=300)

    if test_results.get("check") != "pass":
        errors = []
        for key, value in test_results.items():
            if key.endswith(".error"):
                errors.append(value)
        error_msg = "; ".join(errors[:3]) if errors else test_results.get("error", "Unknown error")
        return {
            "combined_score": 0.0,
            "error": f"Correctness failed: {error_msg}",
            "eval_time": time.time() - start_time,
        }

    # Step 2: Benchmark
    print("[evaluator] Running benchmarks...")
    bench_results = _run_eval(program_path, mode="leaderboard", specs=BENCHMARKS, timeout=600)

    if bench_results.get("check") != "pass":
        errors = []
        for key, value in bench_results.items():
            if key.endswith(".error"):
                errors.append(value)
        error_msg = "; ".join(errors[:3]) if errors else bench_results.get("error", "Unknown error")
        return {
            "combined_score": 0.0,
            "error": f"Benchmark failed: {error_msg}",
            "correctness": "pass",
            "eval_time": time.time() - start_time,
        }

    # Step 3: Extract timings and compute score
    timings_ns = []
    count = int(bench_results.get("benchmark-count", "0"))
    for i in range(count):
        mean_key = f"benchmark.{i}.mean"
        if mean_key in bench_results:
            try:
                timings_ns.append(float(bench_results[mean_key]))
            except ValueError:
                continue

    if not timings_ns:
        return {
            "combined_score": 0.0,
            "error": "No timing data extracted",
            "correctness": "pass",
            "eval_time": time.time() - start_time,
        }

    geomean_ns = _compute_geomean(timings_ns)
    geomean_us = geomean_ns / 1000.0
    score = 1000.0 / geomean_us if geomean_us > 0 else 0.0

    eval_time = time.time() - start_time

    # Collect individual benchmark timings for reporting
    timing_details = {}
    for i in range(count):
        spec = bench_results.get(f"benchmark.{i}.spec", f"benchmark_{i}")
        mean = bench_results.get(f"benchmark.{i}.mean", "N/A")
        timing_details[f"benchmark_{i}_spec"] = spec
        timing_details[f"benchmark_{i}_mean_ns"] = mean

    return {
        "combined_score": score,
        "geomean_us": geomean_us,
        "num_benchmarks": len(timings_ns),
        "correctness": "pass",
        "eval_time": eval_time,
        **timing_details,
    }
