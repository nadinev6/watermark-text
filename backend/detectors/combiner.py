"""
Detection result combination and scoring logic.

This module implements algorithms for combining results from multiple watermark
detectors into unified confidence scores and classifications. It handles
weighting strategies, confidence normalization, and result aggregation.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from detectors.base import DetectionResult, DetectionMethod, DetectionError
from utils.config import get_config
import logging

logger = logging.getLogger(__name__)


@dataclass
class CombinationStrategy:
    """
    Configuration for result combination strategies.
    
    This class defines how multiple detection results should be combined,
    including weighting schemes, confidence thresholds, and aggregation methods.
    
    Attributes:
        synthid_weight (float): Weight for SynthID detector results (0.0-1.0)
        custom_weight (float): Weight for custom detector results (0.0-1.0)
        min_detectors_required (int): Minimum number of detectors needed for valid result
        confidence_threshold (float): Threshold for binary classification
        use_weighted_average (bool): Whether to use weighted or simple averaging
        require_consensus (bool): Whether all detectors must agree on classification
    """
    synthid_weight: float = 0.7
    custom_weight: float = 0.3
    min_detectors_required: int = 1
    confidence_threshold: float = 0.5
    use_weighted_average: bool = True
    require_consensus: bool = False
    
    def __post_init__(self):
        """Validate combination strategy parameters."""
        if abs(self.synthid_weight + self.custom_weight - 1.0) > 0.01:
            raise ValueError("Detector weights must sum to 1.0")
        
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        if self.min_detectors_required < 1:
            raise ValueError("Minimum detectors required must be at least 1")


class ResultCombiner:
    """
    Combines results from multiple watermark detectors into unified outputs.
    
    This class implements various strategies for aggregating detection results,
    including weighted averaging, consensus requirements, and confidence scoring.
    It handles cases where detectors disagree or produce partial results.
    """
    
    def __init__(self, strategy: Optional[CombinationStrategy] = None):
        """
        Initialize the result combiner with specified strategy.
        
        Args:
            strategy (Optional[CombinationStrategy]): Combination strategy to use.
                                                    If None, uses default from config.
        """
        self.config = get_config()
        
        if strategy is None:
            # Create default strategy from configuration
            self.strategy = CombinationStrategy(
                synthid_weight=self.config.detection.synthid_weight,
                custom_weight=self.config.detection.custom_weight,
                confidence_threshold=self.config.detection.default_threshold,
                use_weighted_average=True,
                require_consensus=False
            )
        else:
            self.strategy = strategy
        
        logger.info(f"Initialized ResultCombiner with strategy: {self.strategy}")
    
    def combine_results(
        self,
        results: List[DetectionResult],
        input_text: str
    ) -> DetectionResult:
        """
        Combine multiple detection results into a single unified result.
        
        This method aggregates results from different detectors using the
        configured combination strategy. It handles weighting, normalization,
        and consensus requirements to produce a final classification.
        
        Args:
            results (List[DetectionResult]): List of detection results to combine
            input_text (str): Original input text that was analyzed
            
        Returns:
            DetectionResult: Combined result with aggregated confidence score
                           and unified metadata
                           
        Raises:
            DetectionError: If insufficient results or combination fails
            ValueError: If results list is empty or contains invalid data
        """
        start_time = time.time()
        
        if not results:
            raise ValueError("Cannot combine empty results list")
        
        if len(results) < self.strategy.min_detectors_required:
            raise DetectionError(
                f"Insufficient detectors: got {len(results)}, need {self.strategy.min_detectors_required}",
                "INSUFFICIENT_DETECTORS",
                recoverable=True
            )
        
        try:
            # Calculate combined confidence score
            combined_confidence = self._calculate_combined_confidence(results)
            
            # Determine binary classification
            is_watermarked = self._determine_classification(results, combined_confidence)
            
            # Identify most likely source model
            model_identified = self._identify_source_model(results)
            
            # Aggregate metadata from all detectors
            combined_metadata = self._combine_metadata(results, input_text)
            
            # Calculate total processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            # Create combined result
            combined_result = DetectionResult(
                confidence_score=combined_confidence,
                is_watermarked=is_watermarked,
                model_identified=model_identified,
                detection_method=DetectionMethod.COMBINED.value,
                metadata=combined_metadata,
                processing_time_ms=processing_time
            )
            
            logger.info(
                f"Combined {len(results)} results: confidence={combined_confidence:.3f}, "
                f"watermarked={is_watermarked}, model={model_identified}"
            )
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Failed to combine results: {e}")
            raise DetectionError(
                f"Result combination failed: {str(e)}",
                "COMBINATION_FAILED",
                recoverable=False
            )
    
    def _calculate_combined_confidence(self, results: List[DetectionResult]) -> float:
        """
        Calculate weighted average confidence score from multiple results.
        
        Args:
            results (List[DetectionResult]): Detection results to combine
            
        Returns:
            float: Combined confidence score between 0.0 and 1.0
        """
        if self.strategy.use_weighted_average:
            return self._weighted_average_confidence(results)
        else:
            return self._simple_average_confidence(results)
    
    def _weighted_average_confidence(self, results: List[DetectionResult]) -> float:
        """
        Calculate weighted average confidence based on detector types.
        
        Args:
            results (List[DetectionResult]): Detection results with method information
            
        Returns:
            float: Weighted average confidence score
        """
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = self._get_detector_weight(result.detection_method)
            total_weighted_score += result.confidence_score * weight
            total_weight += weight
        
        if total_weight == 0:
            logger.warning("No valid detector weights found, using simple average")
            return self._simple_average_confidence(results)
        
        return min(1.0, max(0.0, total_weighted_score / total_weight))
    
    def _simple_average_confidence(self, results: List[DetectionResult]) -> float:
        """
        Calculate simple arithmetic average of confidence scores.
        
        Args:
            results (List[DetectionResult]): Detection results to average
            
        Returns:
            float: Simple average confidence score
        """
        total_score = sum(result.confidence_score for result in results)
        return min(1.0, max(0.0, total_score / len(results)))
    
    def _get_detector_weight(self, detection_method: str) -> float:
        """
        Get weight for specific detector type.
        
        Args:
            detection_method (str): Name of the detection method
            
        Returns:
            float: Weight value for this detector type
        """
        method_weights = {
            DetectionMethod.SYNTHID.value: self.strategy.synthid_weight,
            DetectionMethod.CUSTOM.value: self.strategy.custom_weight,
        }
        
        return method_weights.get(detection_method, 0.1)  # Default small weight
    
    def _determine_classification(
        self,
        results: List[DetectionResult],
        combined_confidence: float
    ) -> bool:
        """
        Determine binary watermark classification from results and confidence.
        
        Args:
            results (List[DetectionResult]): Individual detection results
            combined_confidence (float): Combined confidence score
            
        Returns:
            bool: True if text is classified as watermarked
        """
        if self.strategy.require_consensus:
            # All detectors must agree on classification
            classifications = [result.is_watermarked for result in results]
            return all(classifications) if any(classifications) else False
        else:
            # Use confidence threshold for classification
            return combined_confidence >= self.strategy.confidence_threshold
    
    def _identify_source_model(self, results: List[DetectionResult]) -> Optional[str]:
        """
        Identify the most likely source model from detection results.
        
        Args:
            results (List[DetectionResult]): Detection results with model information
            
        Returns:
            Optional[str]: Most likely source model name, or None if unknown
        """
        # Count model identifications weighted by confidence
        model_scores: Dict[str, float] = {}
        
        for result in results:
            if result.model_identified:
                weight = self._get_detector_weight(result.detection_method)
                confidence_weighted_score = result.confidence_score * weight
                
                if result.model_identified in model_scores:
                    model_scores[result.model_identified] += confidence_weighted_score
                else:
                    model_scores[result.model_identified] = confidence_weighted_score
        
        if not model_scores:
            return None
        
        # Return model with highest weighted score
        return max(model_scores.items(), key=lambda x: x[1])[0]
    
    def _combine_metadata(
        self,
        results: List[DetectionResult],
        input_text: str
    ) -> Dict[str, Any]:
        """
        Aggregate metadata from all detection results.
        
        Args:
            results (List[DetectionResult]): Detection results with metadata
            input_text (str): Original input text
            
        Returns:
            Dict[str, Any]: Combined metadata dictionary
        """
        combined_metadata = {
            "combination_strategy": {
                "synthid_weight": self.strategy.synthid_weight,
                "custom_weight": self.strategy.custom_weight,
                "use_weighted_average": self.strategy.use_weighted_average,
                "require_consensus": self.strategy.require_consensus,
                "confidence_threshold": self.strategy.confidence_threshold
            },
            "input_stats": {
                "text_length": len(input_text),
                "word_count": len(input_text.split()),
                "character_count": len(input_text)
            },
            "detector_results": [],
            "combination_stats": {
                "total_detectors": len(results),
                "successful_detectors": len([r for r in results if r.confidence_score >= 0]),
                "consensus_agreement": self._calculate_consensus_agreement(results)
            }
        }
        
        # Add individual detector metadata
        for result in results:
            detector_info = {
                "method": result.detection_method,
                "confidence": result.confidence_score,
                "classification": result.is_watermarked,
                "model_identified": result.model_identified,
                "processing_time_ms": result.processing_time_ms,
                "metadata": result.metadata
            }
            combined_metadata["detector_results"].append(detector_info)
        
        return combined_metadata
    
    def _calculate_consensus_agreement(self, results: List[DetectionResult]) -> float:
        """
        Calculate agreement level between detectors.
        
        Args:
            results (List[DetectionResult]): Detection results to analyze
            
        Returns:
            float: Agreement level between 0.0 (no agreement) and 1.0 (full consensus)
        """
        if len(results) <= 1:
            return 1.0
        
        classifications = [result.is_watermarked for result in results]
        positive_count = sum(classifications)
        negative_count = len(classifications) - positive_count
        
        # Calculate agreement as the proportion of the majority
        majority_count = max(positive_count, negative_count)
        return majority_count / len(classifications)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the current combination strategy.
        
        Returns:
            Dict[str, Any]: Strategy configuration and parameters
        """
        return {
            "synthid_weight": self.strategy.synthid_weight,
            "custom_weight": self.strategy.custom_weight,
            "min_detectors_required": self.strategy.min_detectors_required,
            "confidence_threshold": self.strategy.confidence_threshold,
            "use_weighted_average": self.strategy.use_weighted_average,
            "require_consensus": self.strategy.require_consensus
        }
    
    def update_strategy(self, **kwargs) -> None:
        """
        Update combination strategy parameters.
        
        Args:
            **kwargs: Strategy parameters to update
            
        Raises:
            ValueError: If invalid parameters are provided
        """
        # Create new strategy with updated parameters
        current_params = {
            "synthid_weight": self.strategy.synthid_weight,
            "custom_weight": self.strategy.custom_weight,
            "min_detectors_required": self.strategy.min_detectors_required,
            "confidence_threshold": self.strategy.confidence_threshold,
            "use_weighted_average": self.strategy.use_weighted_average,
            "require_consensus": self.strategy.require_consensus
        }
        
        # Update with provided parameters
        current_params.update(kwargs)
        
        # Validate and create new strategy
        self.strategy = CombinationStrategy(**current_params)
        
        logger.info(f"Updated combination strategy: {self.strategy}")


def create_default_combiner() -> ResultCombiner:
    """
    Create a ResultCombiner with default configuration.
    
    This function creates a combiner instance using the default strategy
    loaded from the application configuration.
    
    Returns:
        ResultCombiner: Configured combiner instance ready for use
    """
    return ResultCombiner()


def create_consensus_combiner() -> ResultCombiner:
    """
    Create a ResultCombiner that requires consensus between detectors.
    
    This function creates a combiner that only classifies text as watermarked
    when all available detectors agree on the classification.
    
    Returns:
        ResultCombiner: Consensus-based combiner instance
    """
    strategy = CombinationStrategy(
        synthid_weight=0.5,
        custom_weight=0.5,
        min_detectors_required=2,
        confidence_threshold=0.5,
        use_weighted_average=True,
        require_consensus=True
    )
    
    return ResultCombiner(strategy)


def create_conservative_combiner() -> ResultCombiner:
    """
    Create a ResultCombiner with conservative classification thresholds.
    
    This function creates a combiner with higher confidence thresholds
    to reduce false positive rates in watermark detection.
    
    Returns:
        ResultCombiner: Conservative combiner instance
    """
    strategy = CombinationStrategy(
        synthid_weight=0.8,
        custom_weight=0.2,
        min_detectors_required=1,
        confidence_threshold=0.75,
        use_weighted_average=True,
        require_consensus=False
    )
    
    return ResultCombiner(strategy)