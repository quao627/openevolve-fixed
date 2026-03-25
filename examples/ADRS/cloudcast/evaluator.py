"""Evaluator for Cloudcast multi-cloud broadcast optimization.

Requires profiles/ and examples/config/ data files in the same directory.
"""
import importlib.util
import traceback
import json
import os
import sys
import tempfile
from pathlib import Path


def evaluate(program_path):
    try:
        # Add evaluator dir and program dir to path BEFORE any imports
        evaluator_dir = os.path.dirname(os.path.abspath(__file__))
        prog_dir = os.path.dirname(os.path.abspath(program_path))
        for p in [evaluator_dir, prog_dir]:
            if p not in sys.path:
                sys.path.insert(0, p)

        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "search_algorithm"):
            return {"combined_score": 0.0, "error": "Missing search_algorithm function"}

        import utils
        import simulator
        import broadcast
        from initial_program import make_nx_graph

        config_files = [
            os.path.join(evaluator_dir, "examples/config/intra_aws.json"),
            os.path.join(evaluator_dir, "examples/config/intra_azure.json"),
            os.path.join(evaluator_dir, "examples/config/intra_gcp.json"),
            os.path.join(evaluator_dir, "examples/config/inter_agz.json"),
            os.path.join(evaluator_dir, "examples/config/inter_gaz2.json"),
        ]
        existing = [f for f in config_files if os.path.exists(f)]
        if not existing:
            return {"combined_score": 0.0, "error": "No config files found"}

        num_vms = 2
        total_cost = 0.0
        successful = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                for jsonfile in existing:
                    with open(jsonfile, "r") as f:
                        config_name = os.path.basename(jsonfile).split(".")[0]
                        config = json.loads(f.read())
                    G = make_nx_graph(num_vms=num_vms)
                    source = config["source_node"]
                    terminals = config["dest_nodes"]
                    num_partitions = config["num_partitions"]

                    bc_t = program.search_algorithm(source, terminals, G, num_partitions)
                    bc_t.set_num_partitions(num_partitions)

                    directory = f"paths/{config_name}"
                    Path(directory).mkdir(parents=True, exist_ok=True)
                    outf = f"{directory}/search_algorithm.json"
                    with open(outf, "w") as out:
                        out.write(json.dumps({
                            "algo": "search_algorithm",
                            "source_node": bc_t.src,
                            "terminal_nodes": bc_t.dsts,
                            "num_partitions": bc_t.num_partitions,
                            "generated_path": bc_t.paths,
                        }))

                    output_dir = f"evals/{config_name}"
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    sim = simulator.BCSimulator(num_vms, output_dir)
                    _, cost = sim.evaluate_path(outf, config)
                    total_cost += cost
                    successful += 1
            finally:
                os.chdir(orig)

        if successful == 0:
            return {"combined_score": 0.0, "error": "All configs failed"}

        combined = 1.0 / (1.0 + total_cost)
        return {"combined_score": combined, "total_cost": total_cost, "successful_configs": successful}

    except Exception as e:
        traceback.print_exc()
        return {"combined_score": 0.0, "error": str(e)}
