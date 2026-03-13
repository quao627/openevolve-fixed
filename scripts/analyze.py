#!/usr/bin/env python3
"""Evolution dynamics analyzer for OpenEvolve runs.

Single-run analysis:
    python scripts/analyze.py --path <output_dir> [--metric sum_radii] [--output analysis.png]

Benchmark summary (multi-problem results from run_all_alphaevolve.sh):
    python scripts/analyze.py --benchmark <results_dir> [--output report.png]
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from numbers import Number

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(v):
    """Return v as float if it's a finite number, else None."""
    if isinstance(v, Number) and not (math.isinf(v) or math.isnan(v)):
        return float(v)
    return None


# ---------------------------------------------------------------------------
# Primary data source: evolution_log.jsonl
# ---------------------------------------------------------------------------

def load_from_evolution_log(log_path, metric):
    """Load per-iteration (iteration, score) pairs from the JSONL log."""
    iterations = []
    scores = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            it = entry.get("iteration")
            score = safe_float(entry.get("metrics", {}).get(metric))
            if it is not None and score is not None:
                iterations.append(it)
                scores.append(score)
    return iterations, scores


# ---------------------------------------------------------------------------
# Fallback: checkpoint scanning (legacy)
# ---------------------------------------------------------------------------

def find_all_checkpoints(base_folder):
    """Return checkpoint dirs sorted by iteration number (ascending)."""
    if os.path.basename(base_folder).startswith("checkpoint_"):
        return [base_folder]

    checkpoints_dir = os.path.join(base_folder, "checkpoints")
    if os.path.isdir(checkpoints_dir):
        search_root = checkpoints_dir
    else:
        search_root = base_folder

    pattern = os.path.join(search_root, "checkpoint_*")
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    def iteration_number(path):
        m = re.search(r"checkpoint_(\d+)$", path)
        return int(m.group(1)) if m else 0

    dirs.sort(key=iteration_number)
    return dirs


def load_all_programs(checkpoint_dir):
    """Load all program JSONs from a checkpoint's programs/ directory."""
    programs_dir = os.path.join(checkpoint_dir, "programs")
    if not os.path.isdir(programs_dir):
        return []

    programs = []
    for fname in os.listdir(programs_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(programs_dir, fname)) as f:
            prog = json.load(f)
        programs.append(prog)
    return programs


def load_from_checkpoints(base_folder, metric):
    """Scan checkpoint program files and return (iterations, scores)."""
    checkpoints = find_all_checkpoints(base_folder)
    if not checkpoints:
        return [], []

    print(f"Found {len(checkpoints)} checkpoint(s)")

    programs_by_id = {}
    for cp in checkpoints:
        for prog in load_all_programs(cp):
            programs_by_id[prog.get("id")] = prog
    programs = list(programs_by_id.values())
    print(f"Loaded {len(programs)} unique programs across all checkpoints")

    iterations = []
    scores = []
    for prog in programs:
        it = prog.get("iteration_found")
        score = safe_float(prog.get("metrics", {}).get(metric))
        if it is None or score is None:
            continue
        iterations.append(it)
        scores.append(score)
    return iterations, scores


# ---------------------------------------------------------------------------
# Benchmark mode: summarize multi-problem results
# ---------------------------------------------------------------------------

def benchmark_report(results_dir, output_path, show=False):
    """Load summary.json from each sub-problem and print/plot a report."""
    summaries = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*/summary.json"))):
        with open(f) as fh:
            summaries.append(json.load(fh))

    if not summaries:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Load run metadata if present
    meta_path = os.path.join(results_dir, "run_config.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # ── Text table ──
    hdr = f"{'Problem':<28} {'Status':<16} {'Time':>8}  {'Score':>12}"
    sep = "─" * len(hdr)
    print()
    if meta:
        print(f"Model: {meta.get('served_model', meta.get('model', '?'))}")
        print(f"Iterations: {meta.get('phase1_iter','?')}+{meta.get('phase2_iter','?')}")
    print(sep)
    print(hdr)
    print(sep)

    problems, scores, statuses = [], [], []
    for s in summaries:
        score = s.get("best_metrics", {}).get("combined_score")
        score_str = f"{score:.6f}" if isinstance(score, (int, float)) else "N/A"
        elapsed = s.get("elapsed_seconds", 0)
        h, m, sec = elapsed // 3600, elapsed % 3600 // 60, elapsed % 60
        time_str = f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"
        print(f"{s['problem']:<28} {s['status']:<16} {time_str:>8}  {score_str:>12}")
        problems.append(s["problem"])
        scores.append(safe_float(score))
        statuses.append(s["status"])

    print(sep)
    completed = sum(1 for st in statuses if st == "completed")
    valid_scores = [s for s in scores if s is not None]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    print(f"Completed: {completed}/{len(summaries)}  |  Avg score: {avg:.4f}")
    print()

    # ── Save consolidated JSON ──
    out_json = os.path.join(results_dir, "all_results.json")
    with open(out_json, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"Saved {out_json}")

    # ── Bar chart ──
    fig, ax = plt.subplots(figsize=(max(10, len(problems) * 0.7), 6))
    colors = [
        "#2ecc71" if st == "completed" else "#e74c3c" for st in statuses
    ]
    bar_scores = [s if s is not None else 0 for s in scores]
    bars = ax.bar(range(len(problems)), bar_scores, color=colors, edgecolor="white")

    # 1.0 reference line (AlphaEvolve parity)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(range(len(problems)))
    ax.set_xticklabels(problems, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("combined_score")
    ax.set_title("AlphaEvolve Benchmark Results", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Single-run analysis
# ---------------------------------------------------------------------------

def single_run_analysis(path, metric, output_path, show=False):
    """Analyze a single OpenEvolve run."""
    if not os.path.isdir(path):
        print(f"Error: {path} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Try the JSONL log first — fast, one-file read
    log_path = os.path.join(path, "evolution_log.jsonl")
    if os.path.isfile(log_path):
        print(f"Reading evolution log: {log_path}")
        all_its, all_scores = load_from_evolution_log(log_path, metric)
        print(f"Loaded {len(all_scores)} iteration entries from log")
    else:
        # Fall back to checkpoint scanning
        print("No evolution_log.jsonl found, falling back to checkpoint scanning")
        all_its, all_scores = load_from_checkpoints(path, metric)

    if not all_scores:
        print(f"Error: no data found for metric '{metric}'", file=sys.stderr)
        sys.exit(1)

    # Compute running max from the actual program data
    paired = sorted(zip(all_its, all_scores))
    best_its = []
    running_max = []
    best_so_far = -float("inf")
    for it, sc in paired:
        if sc > best_so_far:
            best_so_far = sc
            best_its.append(it)
            running_max.append(best_so_far)

    # Single plot: scatter of all programs + best-over-time line
    fig, ax = plt.subplots(figsize=(10, 6))

    if all_its:
        ax.scatter(all_its, all_scores, alpha=0.4, s=12, color="steelblue", label="All programs")

    if best_its:
        ax.step(best_its, running_max, where="post", color="black", linewidth=1.5, label="Best so far")

    ax.set_title(f"Evolution Progress — {metric}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved analysis to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze OpenEvolve runs (single or benchmark)"
    )
    # Mutually exclusive: single-run vs benchmark
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--path", type=str,
        help="Single-run: path to OpenEvolve output directory or checkpoint",
    )
    mode.add_argument(
        "--benchmark", type=str,
        help="Benchmark: path to multi-problem results directory",
    )
    parser.add_argument("--metric", type=str, default="combined_score",
                        help="Metric to plot for single-run mode (default: combined_score)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path")
    parser.add_argument("--show", action="store_true",
                        help="Display figure interactively")
    args = parser.parse_args()

    if args.benchmark:
        out = args.output or os.path.join(args.benchmark, "benchmark_report.png")
        benchmark_report(args.benchmark, out, args.show)
    else:
        out = args.output or "analysis.png"
        single_run_analysis(args.path, args.metric, out, args.show)


if __name__ == "__main__":
    main()
