#!/usr/bin/env python3
"""
Run TriMul kernel engineering evolution using Claude Code CLI (Max plan).

Instead of calling the Anthropic API directly, this uses the `claude` CLI
which authenticates via your Claude Max subscription.

Usage:
    python run_claude_code.py [--iterations N]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add openevolve to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from openevolve.config import load_config
from openevolve.controller import OpenEvolve
from openevolve.llm.claude_code import init_claude_code_client


def main():
    parser = argparse.ArgumentParser(description="Run kernel engineering with Claude Code")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    example_dir = Path(__file__).parent

    # Load config
    config_path = args.config or str(example_dir / "config.yaml")
    config = load_config(config_path)

    # Override iterations if specified
    config.max_iterations = args.iterations

    # Inject Claude Code client into all models
    for model_cfg in config.llm.models:
        model_cfg.init_client = init_claude_code_client
        if not model_cfg.name:
            model_cfg.name = "claude-opus-4-6"

    for model_cfg in config.llm.evaluator_models:
        model_cfg.init_client = init_claude_code_client
        if not model_cfg.name:
            model_cfg.name = "claude-opus-4-6"

    # Create and run OpenEvolve
    initial_program = str(example_dir / "initial_program.py")
    evaluator = str(example_dir / "evaluator.py")

    evolve = OpenEvolve(
        initial_program_path=initial_program,
        evaluation_file=evaluator,
        config=config,
    )

    # Run evolution
    best_program, best_score = asyncio.run(evolve.run())
    print(f"\nBest score: {best_score}")
    print(f"Best program saved to output directory")


if __name__ == "__main__":
    main()
