"""
Watermark detection engines module.

This module provides the core detection infrastructure including abstract
base classes, result combination logic, and detector implementations.
"""

from .base import WatermarkDetector, DetectionResult, DetectionMethod, DetectionError
from .combiner import (
    ResultCombiner,
    CombinationStrategy,
    create_default_combiner,
    create_consensus_combiner,
    create_conservative_combiner
)

# Optional imports that require torch
try:
    from .synthid_detector import SynthIDDetector
    from .custom_detector import CustomDetector, create_custom_detector, create_statistical_detector
    _ml_detectors_available = True
except ImportError:
    SynthIDDetector = None
    CustomDetector = None
    create_custom_detector = None
    create_statistical_detector = None
    _ml_detectors_available = False

# Model identifier doesn't require torch
from .model_identifier import ModelIdentifier, ModelSignature, ModelFamily

__all__ = [
    "WatermarkDetector",
    "DetectionResult", 
    "DetectionMethod",
    "DetectionError",
    "ResultCombiner",
    "CombinationStrategy",
    "create_default_combiner",
    "create_consensus_combiner", 
    "create_conservative_combiner",
    "ModelIdentifier",
    "ModelSignature",
    "ModelFamily",
]

# Only add ML detector exports if available
if _ml_detectors_available:
    __all__.extend([
        "SynthIDDetector",
        "CustomDetector",
        "create_custom_detector",
        "create_statistical_detector"
    ])