"""OpenEvolve evaluator for the Spaceship Titanic classification task.

Evaluates evolved programs by running them against the test set and computing
classification accuracy against the MLEBench answer key.

Metric: accuracy (higher is better) → returned as combined_score.
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
ANSWERS_PATH = os.path.join(DATA_DIR, "answers.csv")

TIMEOUT = 120


def evaluate(program_path: str) -> dict:
    """Evaluate an evolved program on the Spaceship Titanic task.

    Args:
        program_path: Path to the evolved initial_program.py.

    Returns:
        Dict with combined_score (accuracy, 0-1) and metadata.
    """
    try:
        result = _run_in_subprocess(program_path)
    except subprocess.TimeoutExpired:
        return {
            "combined_score": 0.0,
            "accuracy": 0.0,
            "error": f"Timed out after {TIMEOUT}s",
        }
    except Exception as e:
        return {
            "combined_score": 0.0,
            "accuracy": 0.0,
            "error": str(e)[-500:],
        }

    if "error" in result:
        return {
            "combined_score": 0.0,
            "accuracy": 0.0,
            "error": result["error"],
        }

    accuracy = result["accuracy"]
    return {
        "combined_score": accuracy,
        "accuracy": accuracy,
        "n_correct": result["n_correct"],
        "n_total": result["n_total"],
        "eval_time": result.get("eval_time", 0.0),
    }


def _run_in_subprocess(program_path: str) -> dict:
    """Run the evolved program in an isolated subprocess and grade it."""
    script = textwrap.dedent(f"""\
        import json, sys, os, time
        import pandas as pd
        from sklearn.metrics import accuracy_score

        program_path = {os.path.abspath(program_path)!r}
        train_path = {TRAIN_PATH!r}
        test_path = {TEST_PATH!r}
        answers_path = {ANSWERS_PATH!r}

        sys.path.insert(0, os.path.dirname(program_path))
        module_name = os.path.splitext(os.path.basename(program_path))[0]
        spec = __import__("importlib").util.spec_from_file_location(module_name, program_path)
        program = __import__("importlib").util.module_from_spec(spec)
        spec.loader.exec_module(program)

        start = time.time()
        submission = program.run(train_path, test_path)
        eval_time = time.time() - start

        if not isinstance(submission, pd.DataFrame):
            print(json.dumps({{"error": f"run() returned {{type(submission).__name__}}, expected DataFrame"}}))
            sys.exit(0)

        for col in ["PassengerId", "Transported"]:
            if col not in submission.columns:
                print(json.dumps({{"error": f"Missing column: {{col}}"}}))
                sys.exit(0)

        answers = pd.read_csv(answers_path)

        if len(submission) != len(answers):
            print(json.dumps({{"error": f"Got {{len(submission)}} rows, expected {{len(answers)}}"}}))
            sys.exit(0)

        submission = submission.sort_values("PassengerId").reset_index(drop=True)
        answers = answers.sort_values("PassengerId").reset_index(drop=True)

        if (submission["PassengerId"].values != answers["PassengerId"].values).any():
            print(json.dumps({{"error": "PassengerIds do not match answer key"}}))
            sys.exit(0)

        y_true = answers["Transported"].to_numpy()
        y_pred = submission["Transported"].to_numpy()
        acc = float(accuracy_score(y_true=y_true, y_pred=y_pred))
        n_correct = int((y_true == y_pred).sum())

        print(json.dumps({{
            "accuracy": round(acc, 5),
            "n_correct": n_correct,
            "n_total": len(answers),
            "eval_time": round(eval_time, 2),
        }}))
    """)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        return {"error": stderr[-500:] if stderr else "Unknown error"}

    return json.loads(result.stdout)


if __name__ == "__main__":
    # Quick test with the initial program
    initial = os.path.join(os.path.dirname(__file__), "initial_program.py")
    result = evaluate(initial)
    print(json.dumps(result, indent=2))
