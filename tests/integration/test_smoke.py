"""
Smoke tests for integration testing - fast tests that validate basic functionality
These run in CI to ensure core components work without requiring slow LLM calls
"""

import pytest
import tempfile
from pathlib import Path

from openevolve import run_evolution, evolve_function, evolve_code
from openevolve.config import Config, LLMModelConfig
from openevolve.controller import OpenEvolve
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from test_utils import get_evolution_test_program, get_evolution_test_evaluator


def get_mock_config() -> Config:
    """Get config with mock/fast settings for smoke tests"""
    config = Config()
    config.max_iterations = 1
    config.checkpoint_interval = 50
    config.database.in_memory = True
    config.evaluator.cascade_evaluation = False
    config.evaluator.parallel_evaluations = 1
    config.evaluator.timeout = 5  # Very short timeout
    
    # Use empty models list - will trigger validation but won't try to make LLM calls
    config.llm.timeout = 5
    config.llm.retries = 0
    config.llm.models = []
    
    return config


class TestSmoke:
    """Fast smoke tests for CI"""

    def test_controller_initialization(self, test_program_file, test_evaluator_file):
        """Test that OpenEvolve controller can be initialized"""
        config = get_mock_config()
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=config,
            output_dir=tempfile.mkdtemp()
        )
        
        # Test basic initialization
        assert controller is not None
        assert controller.database is not None
        assert controller.evaluator is not None
        assert len(controller.database.programs) == 1  # Initial program loaded

    def test_database_operations(self, test_program_file, test_evaluator_file):
        """Test database operations work correctly"""
        config = get_mock_config()
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=config,
            output_dir=tempfile.mkdtemp()
        )
        
        # Test database functionality
        initial_count = len(controller.database.programs)
        assert initial_count == 1
        
        # Test program retrieval
        program_ids = list(controller.database.programs.keys())
        assert len(program_ids) == 1
        
        first_program = controller.database.get(program_ids[0])
        assert first_program is not None
        assert hasattr(first_program, 'code')
        assert hasattr(first_program, 'metrics')

    def test_evaluator_works(self, test_program_file, test_evaluator_file):
        """Test that evaluator can evaluate the initial program"""
        config = get_mock_config()
        
        controller = OpenEvolve(
            initial_program_path=str(test_program_file),
            evaluation_file=str(test_evaluator_file),
            config=config,
            output_dir=tempfile.mkdtemp()
        )
        
        # The initial program should have been evaluated during initialization
        programs = list(controller.database.programs.values())
        initial_program = programs[0]
        
        assert initial_program.metrics is not None
        assert 'score' in initial_program.metrics
        assert 'combined_score' in initial_program.metrics

    def test_library_api_validation(self):
        """Test library API gives proper error messages when not configured"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(get_evolution_test_program())
            program_file = f.name
        
        def simple_evaluator(path):
            return {"score": 0.5, "combined_score": 0.5}
        
        # Test that library API properly validates LLM configuration
        with pytest.raises(ValueError, match="No LLM models configured"):
            run_evolution(
                initial_program=program_file,
                evaluator=simple_evaluator,
                iterations=1
            )
        
        # Clean up
        Path(program_file).unlink()

    def test_config_validation(self):
        """Test configuration validation works"""
        config = Config()
        
        # Test that default config has proper structure
        assert hasattr(config, 'llm')
        assert hasattr(config, 'database')
        assert hasattr(config, 'evaluator')
        assert hasattr(config, 'prompt')
        
        # Test defaults
        assert config.max_iterations > 0
        assert config.database.in_memory is True
        assert config.llm.retries >= 0


@pytest.fixture
def test_program_file():
    """Create a temporary test program file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(get_evolution_test_program())
        yield Path(f.name)
    Path(f.name).unlink()


@pytest.fixture  
def test_evaluator_file():
    """Create a temporary test evaluator file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(get_evolution_test_evaluator())
        yield Path(f.name)
    Path(f.name).unlink()