"""
Utility functions and helpers module.

This module provides utility functions including configuration management,
test data generation, and other helper functionality.
"""

from .config import get_config, reload_config, AppConfig

# Optional imports that require torch
try:
    from .test_data_generator import (
        TestDataGenerator,
        GenerationParams,
        create_test_generator,
        generate_sample_dataset,
        create_evaluation_prompts
    )
    _test_data_available = True
except ImportError as e:
    # Torch or other ML dependencies not available
    TestDataGenerator = None
    GenerationParams = None
    create_test_generator = None
    generate_sample_dataset = None
    create_evaluation_prompts = None
    _test_data_available = False

__all__ = [
    "get_config",
    "reload_config", 
    "AppConfig",
]

# Only add test data exports if available
if _test_data_available:
    __all__.extend([
        "TestDataGenerator",
        "GenerationParams",
        "create_test_generator",
        "generate_sample_dataset",
        "create_evaluation_prompts"
    ])