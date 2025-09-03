"""
Mock watermark detector for development and testing.

This module provides a lightweight mock detector that simulates
watermark detection without loading heavy ML models. Useful for
development, testing, and when resources are limited.
"""

import time
import random
import logging
from typing import Dict, Any, List

from detectors.base import WatermarkDetector, DetectionResult, DetectionMethod

logger = logging.getLogger(__name__)


class MockDetector(WatermarkDetector):
    """
    Mock watermark detector for development and testing.
    
    This detector provides realistic-looking results without actually
    performing ML inference. It's useful for frontend development,
    API testing, and resource-constrained environments.
    """
    
    def __init__(self, name: str = "Mock Detector"):
        """Initialize the mock detector."""
        super().__init__(name)
        self._is_initialized = True  # Mock is always "ready"
        self.detection_count = 0
        
        # Simulate different text patterns for more realistic results
        self.ai_patterns = [
            "furthermore", "moreover", "additionally", "consequently",
            "in conclusion", "it is important to note", "comprehensive",
            "utilize", "facilitate", "implement", "optimize"
        ]
        
        logger.info(f"Initialized {name}")
    
    def detect(self, text: str) -> DetectionResult:
        """
        Simulate watermark detection with realistic patterns.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            DetectionResult: Simulated detection result
        """
        start_time = time.time()
        
        # Validate input
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        text = text.strip()
        
        if len(text) < 20:
            raise ValueError("Text too short for reliable watermark detection")
        
        # Simulate processing time
        time.sleep(0.1 + random.uniform(0, 0.2))  # 100-300ms
        
        # Calculate "confidence" based on text characteristics
        confidence_score = self._calculate_mock_confidence(text)
        
        # Determine if watermarked based on confidence
        is_watermarked = confidence_score > 0.5
        
        # Mock model identification
        model_identified = None
        if is_watermarked:
            models = ["google/gemma-2-2b", "microsoft/Phi-3-mini-4k-instruct", "meta-llama/Llama-2-7b"]
            model_identified = random.choice(models)
        
        # Create metadata
        metadata = self._create_mock_metadata(text, confidence_score)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        self.detection_count += 1
        
        result = DetectionResult(
            confidence_score=confidence_score,
            is_watermarked=is_watermarked,
            model_identified=model_identified,
            detection_method=DetectionMethod.CUSTOM.value,  # Use CUSTOM for mock
            metadata=metadata,
            processing_time_ms=processing_time
        )
        
        logger.info(
            f"Mock detection completed: confidence={confidence_score:.3f}, "
            f"watermarked={is_watermarked}, time={processing_time}ms"
        )
        
        return result
    
    def _calculate_mock_confidence(self, text: str) -> float:
        """
        Calculate mock confidence score based on text patterns.
        
        Args:
            text (str): Input text
            
        Returns:
            float: Mock confidence score (0.0-1.0)
        """
        # Base score with some randomness
        base_score = random.uniform(0.2, 0.8)
        
        # Adjust based on text characteristics
        text_lower = text.lower()
        
        # Check for AI-like patterns
        ai_pattern_count = sum(1 for pattern in self.ai_patterns if pattern in text_lower)
        ai_pattern_bonus = min(0.3, ai_pattern_count * 0.05)
        
        # Check for repetitive patterns
        words = text_lower.split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_penalty = max(0, (len(words) - unique_words) / len(words) * 0.2)
        else:
            repetition_penalty = 0
        
        # Check for formal language patterns
        formal_indicators = ["therefore", "however", "nevertheless", "furthermore"]
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        formal_bonus = min(0.2, formal_count * 0.1)
        
        # Check text length (longer texts might be more likely AI-generated)
        length_factor = min(0.1, len(text) / 10000)
        
        # Combine factors
        final_score = base_score + ai_pattern_bonus + formal_bonus + length_factor - repetition_penalty
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, final_score))
    
    def _create_mock_metadata(self, text: str, confidence_score: float) -> Dict[str, Any]:
        """
        Create mock metadata for the detection result.
        
        Args:
            text (str): Input text
            confidence_score (float): Calculated confidence score
            
        Returns:
            Dict[str, Any]: Mock metadata
        """
        words = text.split()
        
        return {
            "detector_info": {
                "name": self.name,
                "version": "1.0.0-mock",
                "detection_count": self.detection_count
            },
            "text_analysis": {
                "text_length": len(text),
                "word_count": len(words),
                "unique_words": len(set(word.lower() for word in words)),
                "avg_word_length": sum(len(word) for word in words) / max(1, len(words)),
                "sentence_count": text.count('.') + text.count('!') + text.count('?')
            },
            "detection_scores": {
                "base_score": round(random.uniform(0.2, 0.8), 3),
                "pattern_bonus": round(random.uniform(0, 0.3), 3),
                "formal_bonus": round(random.uniform(0, 0.2), 3),
                "length_factor": round(min(0.1, len(text) / 10000), 3)
            },
            "patterns_detected": {
                "ai_indicators": random.randint(0, 5),
                "formal_language": random.randint(0, 3),
                "repetitive_structures": random.randint(0, 2)
            },
            "mock_info": {
                "is_mock": True,
                "simulated_processing": True,
                "confidence_calculation": "pattern-based simulation"
            }
        }
    
    def is_available(self) -> bool:
        """Mock detector is always available."""
        return True
    
    def get_supported_models(self) -> List[str]:
        """Get list of models this mock detector can simulate."""
        return [
            "google/gemma-2-2b",
            "microsoft/Phi-3-mini-4k-instruct",
            "meta-llama/Llama-2-7b",
            "mock-model-v1"
        ]
    
    def cleanup(self) -> None:
        """Mock cleanup - nothing to clean up."""
        logger.info(f"{self.name} cleanup completed (mock)")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "model_name": "mock-detector-v1",
            "is_loaded": True,
            "load_time_ms": 0,
            "detection_count": self.detection_count,
            "supported_models": self.get_supported_models(),
            "config": {
                "is_mock": True,
                "simulated_inference": True,
                "realistic_patterns": True
            }
        }


class MockSynthIDDetector(MockDetector):
    """Mock SynthID detector specifically."""
    
    def __init__(self):
        super().__init__("Mock SynthID Detector")
    
    def detect(self, text: str) -> DetectionResult:
        """Detect with SynthID-specific patterns."""
        result = super().detect(text)
        
        # Override detection method
        result.detection_method = DetectionMethod.SYNTHID.value
        
        # Add SynthID-specific metadata
        result.metadata["synthid_simulation"] = {
            "perplexity_score": round(random.uniform(0.3, 0.9), 3),
            "token_distribution": round(random.uniform(0.2, 0.8), 3),
            "repetition_patterns": round(random.uniform(0.1, 0.7), 3),
            "sequence_coherence": round(random.uniform(0.4, 0.9), 3)
        }
        
        return result


class MockCustomDetector(MockDetector):
    """Mock custom detector specifically."""
    
    def __init__(self):
        super().__init__("Mock Custom Detector")
    
    def detect(self, text: str) -> DetectionResult:
        """Detect with custom detector patterns."""
        result = super().detect(text)
        
        # Override detection method
        result.detection_method = DetectionMethod.CUSTOM.value
        
        # Add custom detector metadata
        result.metadata["custom_analysis"] = {
            "statistical_score": round(random.uniform(0.2, 0.9), 3),
            "linguistic_patterns": round(random.uniform(0.3, 0.8), 3),
            "style_consistency": round(random.uniform(0.4, 0.9), 3),
            "vocabulary_analysis": round(random.uniform(0.2, 0.7), 3)
        }
        
        return result