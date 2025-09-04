"""
Test utilities for OpenEvolve tests
Provides common functions and constants for consistent testing
"""

import os
import sys
import time
import subprocess
import requests
from typing import Optional
from openai import OpenAI
from openevolve.config import Config, LLMModelConfig

# Standard test model for integration tests - small and fast
TEST_MODEL = "google/gemma-3-270m-it"
DEFAULT_PORT = 8000
DEFAULT_BASE_URL = f"http://localhost:{DEFAULT_PORT}/v1"

def setup_test_env():
    """Set up test environment with local inference"""
    os.environ["OPTILLM_API_KEY"] = "optillm"
    return TEST_MODEL

def get_test_client(base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """Get OpenAI client configured for local optillm"""
    return OpenAI(api_key="optillm", base_url=base_url)

def start_test_server(model: str = TEST_MODEL, port: int = DEFAULT_PORT) -> subprocess.Popen:
    """
    Start optillm server for testing
    Returns the process handle
    """
    # Set environment for local inference
    env = os.environ.copy()
    env["OPTILLM_API_KEY"] = "optillm"
    
    # Start server
    proc = subprocess.Popen([
        "optillm",
        "--model", model,
        "--port", str(port)
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    for _ in range(30):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(1)
    else:
        # Server didn't start in time
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise RuntimeError(f"optillm server failed to start on port {port}")
    
    return proc

def stop_test_server(proc: subprocess.Popen):
    """Stop the test server"""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

def is_server_running(port: int = DEFAULT_PORT) -> bool:
    """Check if optillm server is running on the given port"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_integration_config(port: int = DEFAULT_PORT) -> Config:
    """Get config for integration tests with optillm"""
    config = Config()
    config.max_iterations = 10  # Small for testing
    config.checkpoint_interval = 5
    config.database.in_memory = True
    config.evaluator.parallel_evaluations = 2
    
    # Configure to use optillm server
    base_url = f"http://localhost:{port}/v1"
    config.llm.api_base = base_url
    config.llm.models = [
        LLMModelConfig(
            name=TEST_MODEL,
            api_key="optillm",
            api_base=base_url,
            weight=1.0
        )
    ]
    
    return config

def get_simple_test_messages():
    """Get simple test messages for basic validation"""
    return [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a simple Python function that returns 'hello'."}
    ]

def get_evolution_test_program():
    """Get a simple program for evolution testing"""
    return """# EVOLVE-BLOCK-START
def solve(x):
    return x * 2
# EVOLVE-BLOCK-END
"""

def get_evolution_test_evaluator():
    """Get a simple evaluator for evolution testing"""
    return """def evaluate(program_path):
    return {"score": 0.5, "complexity": 10}
"""