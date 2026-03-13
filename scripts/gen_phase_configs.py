#!/usr/bin/env python3
"""Generate phase-1 and phase-2 YAML configs for a problem.

Reads the problem's existing config.yaml to extract the system_message and
evaluator timeout, then writes two new configs pointing at a local vLLM server.

Usage:
    python scripts/gen_phase_configs.py examples/.../config.yaml output_dir/ \
        --model Qwen/Qwen2.5-7B-Instruct --api-base http://localhost:8000/v1
"""

import argparse
import yaml
import os


PHASE_TEMPLATE = {
    "log_level": "INFO",
    "checkpoint_interval": 5,
    "diff_based_evolution": False,
    "allow_full_rewrites": True,
}

PHASE_SETTINGS = {
    1: {
        "temperature": 0.7,
        "num_top_programs": 3,
        "population_size": 50,
        "archive_size": 25,
        "num_islands": 4,
        "exploitation_ratio": 0.7,
        "migration_interval": 30,
        "cascade_thresholds": [0.3, 0.6],
    },
    2: {
        "temperature": 0.8,
        "num_top_programs": 4,
        "population_size": 60,
        "archive_size": 30,
        "num_islands": 5,
        "exploitation_ratio": 0.55,
        "migration_interval": 25,
        "cascade_thresholds": [0.4, 0.7],
        "max_code_length": 100000,
    },
}

PHASE2_ADDENDUM = (
    "\n\nPHASE 2 INSTRUCTIONS: The evolution has reached a plateau. "
    "Try fundamentally different approaches rather than incremental tweaks. "
    "Consider alternative mathematical formulations, different optimization "
    "strategies, or novel algorithmic ideas."
)


def build_config(base_cfg, phase, model, api_base, iterations):
    s = PHASE_SETTINGS[phase]
    system_msg = base_cfg.get("prompt", {}).get(
        "system_message",
        "You are an expert mathematician and algorithm designer.",
    )
    if phase == 2:
        system_msg += PHASE2_ADDENDUM

    eval_timeout = base_cfg.get("evaluator", {}).get("timeout", 360)

    cfg = {
        **PHASE_TEMPLATE,
        "max_iterations": iterations,
        "llm": {
            "models": [{"name": model, "weight": 1.0}],
            "api_base": api_base,
            "api_key": "EMPTY",
            "temperature": s["temperature"],
            "top_p": 0.95,
            "max_tokens": 8192,
            "timeout": 300,
        },
        "prompt": {
            "system_message": system_msg,
            "num_top_programs": s["num_top_programs"],
            "use_template_stochasticity": True,
        },
        "database": {
            "population_size": s["population_size"],
            "archive_size": s["archive_size"],
            "num_islands": s["num_islands"],
            "elite_selection_ratio": 0.3,
            "exploitation_ratio": s["exploitation_ratio"],
            "migration_interval": s["migration_interval"],
        },
        "evaluator": {
            "timeout": eval_timeout,
            "cascade_evaluation": True,
            "cascade_thresholds": s["cascade_thresholds"],
            "parallel_evaluations": 2,
            "use_llm_feedback": False,
        },
    }
    if "max_code_length" in s:
        cfg["max_code_length"] = s["max_code_length"]
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base_config", help="Path to problem's config.yaml")
    p.add_argument("output_dir", help="Directory to write phase configs into")
    p.add_argument("--model", required=True)
    p.add_argument("--api-base", required=True)
    p.add_argument("--phase1-iter", type=int, default=100)
    p.add_argument("--phase2-iter", type=int, default=200)
    args = p.parse_args()

    with open(args.base_config) as f:
        base = yaml.safe_load(f)

    for phase, iters in [(1, args.phase1_iter), (2, args.phase2_iter)]:
        cfg = build_config(base, phase, args.model, args.api_base, iters)
        out = os.path.join(args.output_dir, f"config_phase_{phase}.yaml")
        with open(out, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
