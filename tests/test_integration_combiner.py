"""
Integration tests for detection result combination with configuration system.

This module tests the integration between the ResultCombiner and the
application configuration system, ensuring proper initialization and
configuration loading.
"""

import pytest
from unittest.mock import patch, MagicMock
from backend.detectors.combiner import ResultCombiner, create_default_combiner
from backend.detectors.base import DetectionResult, DetectionMethod
from backend.utils.config import AppConfig, DetectionConfig


class TestCombinerConfigIntegration:
    """Test integration between combiner and configuration system."""
    
    @patch('backend.detectors.combiner.get_config')
    def test_combiner_uses_config_weights(self, mock_get_config):
        """Test that combiner properly loads weights from configuration."""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.detection.synthid_weight = 0.8
        mock_config.detection.custom_weight = 0.2
        mock_config.detection.default_threshold = 0.6
        mock_get_config.return_value = mock_config
        
        # Create combiner
        combiner = ResultCombiner()
        
        # Verify configuration is used
        assert combiner.strategy.synthid_weight == 0.8
        assert combiner.strategy.custom_weight == 0.2
        assert combiner.strategy.confidence_threshold == 0.6
    
    @patch('backend.detectors.combiner.get_config')
    def test_combiner_with_real_config_structure(self, mock_get_config):
        """Test combiner with realistic configuration structure."""
        # Create realistic config structure
        detection_config = DetectionConfig(
            synthid_weight=0.75,
            custom_weight=0.25,
            default_threshold=0.55
        )
        
        mock_config = MagicMock()
        mock_config.detection = detection_config
        mock_get_config.return_value = mock_config
        
        # Test combiner creation and usage
        combiner = create_default_combiner()
        
        # Create test results
        results = [
            DetectionResult(
                confidence_score=0.9,
                is_watermarked=True,
                model_identified="google/gemma-2-2b",
                detection_method=DetectionMethod.SYNTHID.value,
                metadata={"test": "data"},
                processing_time_ms=100
            ),
            DetectionResult(
                confidence_score=0.4,
                is_watermarked=False,
                model_identified=None,
                detection_method=DetectionMethod.CUSTOM.value,
                metadata={"test": "data2"},
                processing_time_ms=50
            )
        ]
        
        # Combine results
        combined = combiner.combine_results(results, "Test integration text")
        
        # Verify weighted combination: 0.9 * 0.75 + 0.4 * 0.25 = 0.675 + 0.1 = 0.775
        expected_confidence = 0.775
        assert abs(combined.confidence_score - expected_confidence) < 0.01
        
        # Verify classification with custom threshold
        assert combined.is_watermarked is True  # 0.775 > 0.55
        
        # Verify metadata includes configuration info
        assert "combination_strategy" in combined.metadata
        strategy_info = combined.metadata["combination_strategy"]
        assert strategy_info["synthid_weight"] == 0.75
        assert strategy_info["custom_weight"] == 0.25
        assert strategy_info["confidence_threshold"] == 0.55
    
    @patch('backend.detectors.combiner.get_config')
    def test_combiner_strategy_info_matches_config(self, mock_get_config):
        """Test that strategy info reflects actual configuration."""
        mock_config = MagicMock()
        mock_config.detection.synthid_weight = 0.6
        mock_config.detection.custom_weight = 0.4
        mock_config.detection.default_threshold = 0.7
        mock_get_config.return_value = mock_config
        
        combiner = ResultCombiner()
        strategy_info = combiner.get_strategy_info()
        
        assert strategy_info["synthid_weight"] == 0.6
        assert strategy_info["custom_weight"] == 0.4
        assert strategy_info["confidence_threshold"] == 0.7
        assert strategy_info["use_weighted_average"] is True
        assert strategy_info["require_consensus"] is False


if __name__ == "__main__":
    pytest.main([__file__])