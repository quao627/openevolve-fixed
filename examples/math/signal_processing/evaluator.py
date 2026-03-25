"""Evaluator for adaptive signal processing."""
import importlib.util
import numpy as np
import time
import traceback
import json

def evaluate(program_path):
    try:
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        if not hasattr(program, "run_signal_processing"):
            return {"combined_score": 0.0, "error": "Missing run_signal_processing"}

        np.random.seed(42)
        test_signals = []
        # Generate 5 diverse test signals
        for t_type in range(5):
            n = 500
            t = np.linspace(0, 10, n)
            if t_type == 0:
                clean = np.sin(2*np.pi*t) + 0.5*np.sin(6*np.pi*t)
            elif t_type == 1:
                clean = np.where(t < 5, np.sin(2*np.pi*t), np.sin(10*np.pi*t))
            elif t_type == 2:
                clean = np.cumsum(np.random.randn(n)) / np.sqrt(n)
            elif t_type == 3:
                clean = np.sin(2*np.pi*t*(1+t/10))
            else:
                clean = np.sign(np.sin(2*np.pi*t))
            noise = np.random.randn(n) * 0.5
            test_signals.append((clean, clean + noise))

        scores = []
        for clean, noisy in test_signals:
            try:
                result = program.run_signal_processing(noisy, window_size=20)
                filtered = np.array(result.get("filtered_signal", []))
                if len(filtered) == 0:
                    scores.append(0.0); continue
                # Align lengths - filter may produce shorter output due to windowing
                min_len = min(len(filtered), len(clean))
                filtered = filtered[:min_len]
                clean_trimmed = clean[:min_len]
                noisy_trimmed = noisy[:min_len]
                corr = np.corrcoef(clean_trimmed, filtered)[0, 1] if np.std(filtered) > 0 else 0
                mse = np.mean((clean_trimmed - filtered) ** 2)
                noise_var = np.mean((clean_trimmed - noisy_trimmed) ** 2)
                noise_red = 1 - mse / noise_var if noise_var > 0 else 0
                scores.append(max(0, 0.5 * max(0, corr) + 0.3 * max(0, noise_red) + 0.2))
            except:
                scores.append(0.0)

        combined = float(np.mean(scores)) if scores else 0.0
        return {"combined_score": combined, "per_signal_scores": [float(s) for s in scores]}
    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}
