"""
Abstract base classes and interfaces for watermark detection systems.

This module defines the core interfaces that all watermark detectors must implement,
providing a consistent API for different detection methods including SynthID and
custom fallback algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum


class DetectionMethod(Enum):
    """Enumeration of available watermark detection methods."""
    SYNTHID = "synthid"
    CUSTOM = "custom"
    COMBINED = "combined"


@dataclass
class DetectionResult:
    """
    Result of watermark detection analysis.
    
    This class encapsulates all information returned by a watermark detector,
    including confidence scores, model identification, and metadata about
    the detection process.
    
    Attributes:
        confidence_score (float): Confidence level (0.0-1.0) that text is watermarked
        is_watermarked (bool): Binary classification result
        model_identified (Optional[str]): Name of the model used to generate the text
        detection_method (str): Method used for detection (synthid, custom, combined)
        metadata (Dict[str, Any]): Additional information about the detection process
        processing_time_ms (int): Time taken for detection in milliseconds
    """
    confidence_score: float
    is_watermarked: bool
    model_identified: Optional[str]
    detection_method: str
    metadata: Dict[str, Any]
    processing_time_ms: int = 0
    
    def __post_init__(self):
        """Validate detection result data after initialization."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        
        if self.processing_time_ms < 0:
            raise ValueError("Processing time cannot be negative")


class WatermarkDetector(ABC):
    """
    Abstract base class for all watermark detection implementations.
    
    This class defines the interface that all watermark detectors must implement,
    ensuring consistent behavior across different detection methods. Subclasses
    should implement the detect() method with their specific detection logic.
    """
    
    def __init__(self, name: str):
        """
        Initialize the watermark detector.
        
        Args:
            name (str): Human-readable name for this detector
        """
        self.name = name
        self._is_initialized = False
    
    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        """
        Analyze text for watermark presence.
        
        This method performs the core watermark detection logic and returns
        a structured result with confidence scores and metadata.
        
        Args:
            text (str): Input text to analyze for watermarks
            
        Returns:
            DetectionResult: Structured result containing confidence score,
                           classification, and detection metadata
                           
        Raises:
            ValueError: If input text is empty or invalid
            RuntimeError: If detector is not properly initialized
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the detector is ready for use.
        
        This method verifies that all required models and dependencies
        are loaded and the detector can perform analysis.
        
        Returns:
            bool: True if detector is ready, False otherwise
        """
        pass
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of models this detector can identify.
        
        Returns:
            List[str]: List of model names this detector can identify
        """
        return []
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get detector metadata and configuration information.
        
        Returns:
            Dict[str, Any]: Metadata about detector capabilities and configuration
        """
        return {
            "name": self.name,
            "initialized": self._is_initialized,
            "supported_models": self.get_supported_models()
        }


class DetectionError(Exception):
    """
    Custom exception for watermark detection errors.
    
    This exception is raised when detection operations fail due to
    model loading issues, processing errors, or invalid input.
    
    Attributes:
        message (str): Human-readable error description
        error_code (str): Machine-readable error identifier
        recoverable (bool): Whether the error can be recovered from
    """
    
    def __init__(self, message: str, error_code: str, recoverable: bool = True):
        """
        Initialize detection error.
        
        Args:
            message (str): Descriptive error message
            error_code (str): Unique error code for programmatic handling
            recoverable (bool): Whether operation can be retried
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.recoverable = recoverable