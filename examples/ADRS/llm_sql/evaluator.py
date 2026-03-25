"""Evaluator for LLM-SQL column reordering for prefix caching.

Requires solver.py, utils.py, and datasets/ in the same directory.
"""
import importlib.util
import os
import sys
import time
import traceback

def evaluate(program_path):
    evaluator_dir = os.path.dirname(os.path.abspath(__file__))
    prog_dir = os.path.dirname(os.path.abspath(program_path))
    for p in [evaluator_dir, prog_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        if not hasattr(program, "Evolved"):
            return {"combined_score": 0.0, "error": "Missing Evolved class"}

        import pandas as pd
        from solver import Algorithm

        datasets_dir = os.path.join(evaluator_dir, "datasets")
        if not os.path.exists(datasets_dir):
            return {"combined_score": 0.0, "error": "datasets/ directory not found"}

        csv_files = [f for f in os.listdir(datasets_dir) if f.endswith(".csv")]
        if not csv_files:
            return {"combined_score": 0.0, "error": "No CSV files in datasets/"}

        hit_rates = []
        runtimes = []
        for csv_file in csv_files[:5]:
            try:
                df = pd.read_csv(os.path.join(datasets_dir, csv_file)).fillna("").astype(str)
                if len(df) < 2: continue
                algo = program.Evolved(df)
                start = time.time()
                reordered_df, orderings = algo.reorder(df)
                runtime = time.time() - start
                runtimes.append(runtime)

                # Compute prefix hit rate
                total_hits = 0
                total_possible = 0
                for i in range(1, len(reordered_df)):
                    row_order = orderings[i] if i < len(orderings) else list(reordered_df.columns)
                    hits = 0
                    for col in row_order:
                        if col in reordered_df.columns and str(reordered_df.iloc[i][col]) == str(reordered_df.iloc[i-1][col]):
                            hits += len(str(reordered_df.iloc[i][col])) ** 2
                        else:
                            break
                    total_hits += hits
                    total_possible += sum(len(str(reordered_df.iloc[i][c])) ** 2 for c in row_order if c in reordered_df.columns)
                hit_rate = total_hits / total_possible if total_possible > 0 else 0
                hit_rates.append(hit_rate)
            except Exception as e:
                print(f"Error on {csv_file}: {e}")
                hit_rates.append(0.0)
                runtimes.append(12.0)

        if not hit_rates:
            return {"combined_score": 0.0, "error": "No successful evaluations"}

        avg_hit = sum(hit_rates) / len(hit_rates)
        avg_runtime = sum(runtimes) / len(runtimes)
        speed_score = (12 - min(12, avg_runtime)) / 12
        combined = 0.95 * avg_hit + 0.05 * speed_score
        return {"combined_score": float(combined), "avg_hit_rate": float(avg_hit),
                "avg_runtime": float(avg_runtime), "speed_score": float(speed_score)}

    except Exception as e:
        traceback.print_exc()
        return {"combined_score": 0.0, "error": str(e)}
