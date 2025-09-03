"""
API module for the AI Watermark Detection Tool.

This module provides FastAPI endpoints, middleware, and services for
watermark detection and test data generation functionality.
"""

from .detection import router as detection_router
from .test_data import router as test_data_router
from .watermark import router as watermark_router
from .middleware import setup_middleware
from .dependencies import get_current_config, validate_api_key

__all__ = [
    "detection_router",
    "test_data_router", 
    "watermark_router",
    "setup_middleware",
    "get_current_config",
    "validate_api_key"
]