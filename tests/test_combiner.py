"""
Unit tests for detection result combination logic.

This module tests the ResultCombiner class and related functionality,
including weighted averaging, consensus requirements, and metadata aggregation.
"""

import pytest
from unittest.mock import patch, MagicMock
from backend.detectors.base import DetectionResult, DetectionMethod, DetectionError
from backend.detectors.combiner import (
    ResultCombiner,
    CombinationStrategy,
    create_default_combiner,
    create_consensus_combiner,
    create_conservative_combiner
)


class TestCombinationStrategy:
    """Test cases for CombinationStrategy configuration."""
    
    def test_valid_strategy_creation(self):
        """Test creating a valid combination strategy."""
        strategy = CombinationStrategy(
            synthid_weight=0.6,
            custom_weight=0.4,
            confidence_threshold=0.7
        )
        
        assert strategy.synthid_weight == 0.6
        assert strategy.custom_weight == 0.4
        assert strategy.confidence_threshold == 0.7
    
    def test_invalid_weights_sum(self):
        """Test that invalid weight sums raise ValueError."""
        with pytest.raises(ValueError, match="Detector weights must sum to 1.0"):
            CombinationStrategy(synthid_weight=0.6, custom_weight=0.5)
    
    def test_invalid_confidence_threshold(self):
        """Test that invalid confidence thresholds raise ValueError."""
        with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
            CombinationStrategy(confidence_threshold=1.5)
    
    def test_invalid_min_detectors(self):
        """Test that invalid minimum detector count raises ValueError."""
        with pytest.raises(ValueError, match="Minimum detectors required must be at least 1"):
            CombinationStrategy(min_detectors_required=0)


class TestResultCombiner:
    """Test cases for ResultCombiner functionality."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample detection results for testing."""
        return [
            DetectionResult(
                confidence_score=0.8,
                is_watermarked=True,
                model_identified="google/gemma-2-2b",
                detection_method=DetectionMethod.SYNTHID.value,
                metadata={"synthid_score": 0.8},
                processing_time_ms=100
            ),
            DetectionResult(
                confidence_score=0.6,
                is_watermarked=True,
                model_identified="google/gemma-2-2b",
                detection_method=DetectionMethod.CUSTOM.value,
                metadata={"perplexity": 15.2},
                processing_time_ms=50
            )
        ]
    
    @pytest.fixture
    def conflicting_results(self):
        """Create conflicting detection results for testing."""
        return [
            DetectionResult(
                confidence_score=0.8,
                is_watermarked=True,
                model_identified="google/gemma-2-2b",
                detection_method=DetectionMethod.SYNTHID.value,
                metadata={"synthid_score": 0.8},
                processing_time_ms=100
            ),
            DetectionResult(
                confidence_score=0.3,
                is_watermarked=False,
                model_identified=None,
                detection_method=DetectionMethod.CUSTOM.value,
                metadata={"perplexity": 8.5},
                processing_time_ms=50
            )
        ]
    
    @patch('backend.detectors.combiner.get_config')
    def test_combiner_initialization(self, mock_config):
        """Test ResultCombiner initialization with default strategy."""
        mock_config.return_value.detection.synthid_weight = 0.7
        mock_config.return_value.detection.custom_weight = 0.3
        mock_config.return_value.detection.default_threshold = 0.5
        
        combiner = ResultCombiner()
        
        assert combiner.strategy.synthid_weight == 0.7
        assert combiner.strategy.custom_weight == 0.3
        assert combiner.strategy.confidence_threshold == 0.5
    
    def test_combiner_with_custom_strategy(self):
        """Test ResultCombiner initialization with custom strategy."""
        strategy = CombinationStrategy(
            synthid_weight=0.8,
            custom_weight=0.2,
            confidence_threshold=0.6
        )
        
        combiner = ResultCombiner(strategy)
        
        assert combiner.strategy.synthid_weight == 0.8
        assert combiner.strategy.custom_weight == 0.2
        assert combiner.strategy.confidence_threshold == 0.6
    
    @patch('backend.detectors.combiner.get_config')
    def test_combine_results_weighted_average(self, mock_config, sample_results):
        """Test combining results using weighted average."""
        mock_config.return_value.detection.synthid_weight = 0.7
        mock_config.return_value.detection.custom_weight = 0.3
        mock_config.return_value.detection.default_threshold = 0.5
        
        combiner = ResultCombiner()
        combined = combiner.combine_results(sample_results, "Test text")
        
        # Expected: 0.8 * 0.7 + 0.6 * 0.3 = 0.56 + 0.18 = 0.74
        expected_confidence = 0.74
        
        assert abs(combined.confidence_score - expected_confidence) < 0.01
        assert combined.is_watermarked is True
        assert combined.model_identified == "google/gemma-2-2b"
        assert combined.detection_method == DetectionMethod.COMBINED.value
    
    @patch('backend.detectors.combiner.get_config')
    def test_combine_results_simple_average(self, mock_config, sample_results):
        """Test combining results using simple average."""
        mock_config.return_value.detection.synthid_weight = 0.7
        mock_config.return_value.detection.custom_weight = 0.3
        mock_config.return_value.detection.default_threshold = 0.5
        
        strategy = CombinationStrategy(
            synthid_weight=0.7,
            custom_weight=0.3,
            use_weighted_average=False
        )
        
        combiner = ResultCombiner(strategy)
        combined = combiner.combine_results(sample_results, "Test text")
        
        # Expected: (0.8 + 0.6) / 2 = 0.7
        expected_confidence = 0.7
        
        assert abs(combined.confidence_score - expected_confidence) < 0.01
        assert combined.is_watermarked is True
    
    @patch('backend.detectors.combiner.get_config')
    def test_combine_conflicting_results(self, mock_config, conflicting_results):
        """Test combining conflicting detection results."""
        mock_config.return_value.detection.synthid_weight = 0.7
        mock_config.return_value.detection.custom_weight = 0.3
        mock_config.return_value.detection.default_threshold = 0.5
        
        combiner = ResultCombiner()
        combined = combiner.combine_results(conflicting_results, "Test text")
        
        # Expected: 0.8 * 0.7 + 0.3 * 0.3 = 0.56 + 0.09 = 0.65
        expected_confidence = 0.65
        
        assert abs(combined.confidence_score - expected_confidence) < 0.01
        assert combined.is_watermarked is True  # Above 0.5 threshold
        assert combined.model_identified == "google/gemma-2-2b"
    
    @patch('backend.detectors.combiner.get_config')
    def test_consensus_requirement(self, mock_config, conflicting_results):
        """Test consensus requirement with conflicting results."""
        mock_config.return_value.detection.synthid_weight = 0.5
        mock_config.return_value.detection.custom_weight = 0.5
        mock_config.return_value.detection.default_threshold = 0.5
        
        strategy = CombinationStrategy(
            synthid_weight=0.5,
            custom_weight=0.5,
            require_consensus=True
        )
        
        combiner = ResultCombiner(strategy)
        combined = combiner.combine_results(conflicting_results, "Test text")
        
        # Should be False due to consensus requirement with conflicting results
        assert combined.is_watermarked is False
    
    def test_empty_results_error(self):
        """Test that empty results list raises ValueError."""
        combiner = create_default_combiner()
        
        with pytest.raises(ValueError, match="Cannot combine empty results list"):
            combiner.combine_results([], "Test text")
    
    @patch('backend.detectors.combiner.get_config')
    def test_insufficient_detectors_error(self, mock_config):
        """Test error when insufficient detectors are available."""
        mock_config.return_value.detection.synthid_weight = 0.7
        mock_config.return_value.detection.custom_weight = 0.3
        mock_config.return_value.detection.default_threshold = 0.5
        
        strategy = CombinationStrategy(
            synthid_weight=0.7,
            custom_weight=0.3,
            min_detectors_required=2
        )
        
        combiner = ResultCombiner(strategy)
        
        single_result = [DetectionResult(
            confidence_score=0.8,
            is_watermarked=True,
            model_identified="test",
            detection_method=DetectionMethod.SYNTHID.value,
            metadata={},
            processing_time_ms=100
        )]
        
        with pytest.raises(DetectionError, match="Insufficient detectors"):
            combiner.combine_results(single_result, "Test text")
    
    @patch('backend.detectors.combiner.get_config')
    def test_metadata_combination(self, mock_config, sample_results):
        """Test that metadata is properly combined from all detectors."""
        mock_config.return_value.detection.synthid_weight = 0.7
        mock_config.return_value.detection.custom_weight = 0.3
        mock_config.return_value.detection.default_threshold = 0.5
        
        combiner = ResultCombiner()
        combined = combiner.combine_results(sample_results, "Test text for analysis")
        
        metadata = combined.metadata
        
        # Check combination strategy metadata
        assert "combination_strategy" in metadata
        assert metadata["combination_strategy"]["synthid_weight"] == 0.7
        
        # Check input statistics
        assert "input_stats" in metadata
        assert metadata["input_stats"]["text_length"] == len("Test text for analysis")
        
        # Check detector results
        assert "detector_results" in metadata
        assert len(metadata["detector_results"]) == 2
        
        # Check combination statistics
        assert "combination_stats" in metadata
        assert metadata["combination_stats"]["total_detectors"] == 2
        assert metadata["combination_stats"]["successful_detectors"] == 2
    
    @patch('backend.detectors.combiner.get_config')
    def test_consensus_agreement_calculation(self, mock_config):
        """Test consensus agreement calculation."""
        mock_config.return_value.detection.synthid_weight = 0.5
        mock_config.return_value.detection.custom_weight = 0.5
        mock_config.return_value.detection.default_threshold = 0.5
        
        combiner = ResultCombiner()
        
        # Test full agreement
        agreeing_results = [
            DetectionResult(0.8, True, "model1", "synthid", {}, 100),
            DetectionResult(0.7, True, "model1", "custom", {}, 50)
        ]
        
        agreement = combiner._calculate_consensus_agreement(agreeing_results)
        assert agreement == 1.0
        
        # Test partial agreement (2 out of 3 agree)
        partial_results = [
            DetectionResult(0.8, True, "model1", "synthid", {}, 100),
            DetectionResult(0.7, True, "model1", "custom", {}, 50),
            DetectionResult(0.3, False, None, "custom", {}, 75)
        ]
        
        agreement = combiner._calculate_consensus_agreement(partial_results)
        assert abs(agreement - 0.667) < 0.01  # 2/3 ≈ 0.667
    
    def test_strategy_update(self):
        """Test updating combination strategy parameters."""
        combiner = create_default_combiner()
        
        original_threshold = combiner.strategy.confidence_threshold
        
        combiner.update_strategy(confidence_threshold=0.8)
        
        assert combiner.strategy.confidence_threshold == 0.8
        assert combiner.strategy.confidence_threshold != original_threshold
    
    def test_invalid_strategy_update(self):
        """Test that invalid strategy updates raise ValueError."""
        combiner = create_default_combiner()
        
        with pytest.raises(ValueError):
            combiner.update_strategy(synthid_weight=0.8, custom_weight=0.8)  # Sum > 1.0


class TestCombinerFactories:
    """Test cases for combiner factory functions."""
    
    @patch('backend.detectors.combiner.get_config')
    def test_create_default_combiner(self, mock_config):
        """Test creating default combiner."""
        mock_config.return_value.detection.synthid_weight = 0.7
        mock_config.return_value.detection.custom_weight = 0.3
        mock_config.return_value.detection.default_threshold = 0.5
        
        combiner = create_default_combiner()
        
        assert isinstance(combiner, ResultCombiner)
        assert combiner.strategy.synthid_weight == 0.7
        assert combiner.strategy.use_weighted_average is True
        assert combiner.strategy.require_consensus is False
    
    def test_create_consensus_combiner(self):
        """Test creating consensus-based combiner."""
        combiner = create_consensus_combiner()
        
        assert isinstance(combiner, ResultCombiner)
        assert combiner.strategy.require_consensus is True
        assert combiner.strategy.min_detectors_required == 2
        assert combiner.strategy.synthid_weight == 0.5
        assert combiner.strategy.custom_weight == 0.5
    
    def test_create_conservative_combiner(self):
        """Test creating conservative combiner."""
        combiner = create_conservative_combiner()
        
        assert isinstance(combiner, ResultCombiner)
        assert combiner.strategy.confidence_threshold == 0.75
        assert combiner.strategy.synthid_weight == 0.8
        assert combiner.strategy.custom_weight == 0.2
        assert combiner.strategy.require_consensus is False


if __name__ == "__main__":
    pytest.main([__file__])