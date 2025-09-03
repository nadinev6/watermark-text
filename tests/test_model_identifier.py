"""
Unit tests for model identification functionality.

This module tests the ModelIdentifier class and related functionality
for identifying source models based on detection patterns and metadata.
"""

import pytest
from unittest.mock import patch, MagicMock
from backend.detectors.model_identifier import (
    ModelIdentifier,
    ModelSignature,
    ModelFamily
)


class TestModelSignature:
    """Test cases for ModelSignature dataclass."""
    
    def test_model_signature_creation(self):
        """Test creating a valid model signature."""
        signature = ModelSignature(
            model_name="test/model",
            family=ModelFamily.GEMMA,
            typical_perplexity_range=(5.0, 20.0),
            token_distribution_entropy=6.5,
            repetition_patterns={"immediate": 0.02},
            sequence_coherence=7.0
        )
        
        assert signature.model_name == "test/model"
        assert signature.family == ModelFamily.GEMMA
        assert signature.typical_perplexity_range == (5.0, 20.0)
        assert signature.confidence_threshold == 0.7  # Default value


class TestModelIdentifier:
    """Test cases for ModelIdentifier functionality."""
    
    @pytest.fixture
    def identifier(self):
        """Create ModelIdentifier instance for testing."""
        return ModelIdentifier()
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample detection metadata for testing."""
        return {
            "detection_scores": {
                "perplexity": 0.7,
                "token_distribution": 0.6,
                "repetition_patterns": 0.5,
                "sequence_coherence": 0.8
            },
            "text_analysis": {
                "token_count": 50,
                "unique_tokens": 35,
                "avg_token_length": 4.2
            }
        }
    
    def test_identifier_initialization(self, identifier):
        """Test ModelIdentifier initialization."""
        assert len(identifier.model_signatures) > 0
        assert "google/gemma-2-2b" in identifier.model_signatures
        assert len(identifier.identification_history) == 0
    
    def test_get_supported_models(self, identifier):
        """Test getting list of supported models."""
        models = identifier.get_supported_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "google/gemma-2-2b" in models
        assert "microsoft/Phi-3-mini-4k-instruct" in models
    
    def test_get_model_families(self, identifier):
        """Test getting list of model families."""
        families = identifier.get_model_families()
        
        assert isinstance(families, list)
        assert "gemma" in families
        assert "gpt" in families
    
    def test_extract_analysis_metrics(self, identifier, sample_metadata):
        """Test extracting analysis metrics from metadata."""
        metrics = identifier._extract_analysis_metrics(sample_metadata)
        
        assert metrics is not None
        assert "perplexity" in metrics
        assert "token_distribution_entropy" in metrics
        assert "repetition_score" in metrics
        assert "sequence_coherence" in metrics
        assert "token_count" in metrics
        assert "unique_token_ratio" in metrics
        assert "avg_token_length" in metrics
        
        # Check calculated unique token ratio
        expected_ratio = 35 / 50  # unique_tokens / token_count
        assert abs(metrics["unique_token_ratio"] - expected_ratio) < 0.01
    
    def test_extract_analysis_metrics_missing_data(self, identifier):
        """Test extracting metrics with missing data."""
        incomplete_metadata = {
            "detection_scores": {"perplexity": 0.5},
            "text_analysis": {"token_count": 0}  # Invalid token count
        }
        
        metrics = identifier._extract_analysis_metrics(incomplete_metadata)
        
        assert metrics is None
    
    def test_extract_analysis_metrics_empty(self, identifier):
        """Test extracting metrics from empty metadata."""
        empty_metadata = {}
        
        metrics = identifier._extract_analysis_metrics(empty_metadata)
        
        assert metrics is None
    
    def test_calculate_unique_token_ratio(self, identifier):
        """Test unique token ratio calculation."""
        # Normal case
        text_analysis = {"token_count": 100, "unique_tokens": 75}
        ratio = identifier._calculate_unique_token_ratio(text_analysis)
        assert ratio == 0.75
        
        # Zero tokens case
        text_analysis = {"token_count": 0, "unique_tokens": 0}
        ratio = identifier._calculate_unique_token_ratio(text_analysis)
        assert ratio == 0.0
        
        # Missing data case
        text_analysis = {}
        ratio = identifier._calculate_unique_token_ratio(text_analysis)
        assert ratio == 0.0
    
    def test_calculate_perplexity_similarity(self, identifier):
        """Test perplexity similarity calculation."""
        # Perfect match (within range)
        similarity = identifier._calculate_perplexity_similarity(0.8, (10.0, 30.0))
        assert similarity > 0.5  # Should be high similarity
        
        # Outside range
        similarity = identifier._calculate_perplexity_similarity(0.1, (10.0, 30.0))
        assert 0.0 <= similarity <= 1.0
    
    def test_calculate_entropy_similarity(self, identifier):
        """Test entropy similarity calculation."""
        # Close match
        similarity = identifier._calculate_entropy_similarity(0.65, 6.5)
        assert similarity > 0.8  # Should be high similarity
        
        # Distant match
        similarity = identifier._calculate_entropy_similarity(0.1, 6.5)
        assert similarity < 0.5  # Should be low similarity
    
    def test_calculate_repetition_similarity(self, identifier):
        """Test repetition pattern similarity calculation."""
        expected_patterns = {
            "immediate_repetition": 0.02,
            "bigram_repetition": 0.15,
            "trigram_repetition": 0.08
        }
        
        # Close to average
        avg_expected = sum(expected_patterns.values()) / len(expected_patterns)
        similarity = identifier._calculate_repetition_similarity(avg_expected, expected_patterns)
        assert similarity > 0.8  # Should be high similarity
        
        # Far from average
        similarity = identifier._calculate_repetition_similarity(0.9, expected_patterns)
        assert similarity < 0.5  # Should be low similarity
    
    def test_calculate_coherence_similarity(self, identifier):
        """Test coherence similarity calculation."""
        # Close match
        similarity = identifier._calculate_coherence_similarity(0.9, 7.2)  # 0.9 * 8 ≈ 7.2
        assert similarity > 0.8  # Should be high similarity
        
        # Distant match
        similarity = identifier._calculate_coherence_similarity(0.1, 7.2)
        assert similarity < 0.5  # Should be low similarity
    
    def test_calculate_model_similarity(self, identifier):
        """Test overall model similarity calculation."""
        # Create test metrics
        metrics = {
            "perplexity": 0.7,
            "token_distribution_entropy": 0.65,
            "repetition_score": 0.08,
            "sequence_coherence": 0.9,
            "token_count": 50,
            "unique_token_ratio": 0.7,
            "avg_token_length": 4.0
        }
        
        # Get Gemma signature for testing
        signature = identifier.model_signatures["google/gemma-2-2b"]
        
        similarity = identifier._calculate_model_similarity(
            metrics, signature, 0.8, 200
        )
        
        assert 0.0 <= similarity <= 1.0
    
    def test_calculate_model_similarity_short_text(self, identifier):
        """Test model similarity with short text penalty."""
        metrics = {
            "perplexity": 0.8,
            "token_distribution_entropy": 0.7,
            "repetition_score": 0.05,
            "sequence_coherence": 0.9,
            "token_count": 20,
            "unique_token_ratio": 0.8,
            "avg_token_length": 4.0
        }
        
        signature = identifier.model_signatures["google/gemma-2-2b"]
        
        # Short text (50 characters)
        similarity_short = identifier._calculate_model_similarity(
            metrics, signature, 0.8, 50
        )
        
        # Long text (200 characters)
        similarity_long = identifier._calculate_model_similarity(
            metrics, signature, 0.8, 200
        )
        
        # Short text should have lower similarity due to penalty
        assert similarity_short < similarity_long
    
    def test_identify_model_success(self, identifier, sample_metadata):
        """Test successful model identification."""
        model_name, confidence = identifier.identify_model(
            sample_metadata, 0.8, 200
        )
        
        # Should identify some model with reasonable confidence
        if model_name is not None:
            assert model_name in identifier.get_supported_models()
            assert 0.0 <= confidence <= 1.0
    
    def test_identify_model_insufficient_metadata(self, identifier):
        """Test model identification with insufficient metadata."""
        insufficient_metadata = {
            "detection_scores": {},
            "text_analysis": {"token_count": 0}
        }
        
        model_name, confidence = identifier.identify_model(
            insufficient_metadata, 0.5, 100
        )
        
        assert model_name is None
        assert confidence == 0.0
    
    def test_identify_model_low_confidence(self, identifier):
        """Test model identification with low confidence scores."""
        # Create metadata that should result in low confidence
        low_confidence_metadata = {
            "detection_scores": {
                "perplexity": 0.1,
                "token_distribution": 0.1,
                "repetition_patterns": 0.1,
                "sequence_coherence": 0.1
            },
            "text_analysis": {
                "token_count": 50,
                "unique_tokens": 25,
                "avg_token_length": 3.0
            }
        }
        
        model_name, confidence = identifier.identify_model(
            low_confidence_metadata, 0.2, 100
        )
        
        # Should not identify model due to low confidence
        assert model_name is None or confidence < 0.7
    
    def test_record_identification(self, identifier):
        """Test recording identification results."""
        initial_count = len(identifier.identification_history)
        
        metrics = {"perplexity": 0.8, "token_count": 50}
        all_scores = {"google/gemma-2-2b": 0.85, "microsoft/Phi-3-mini-4k-instruct": 0.6}
        
        identifier._record_identification(
            "google/gemma-2-2b", 0.85, metrics, all_scores
        )
        
        assert len(identifier.identification_history) == initial_count + 1
        
        record = identifier.identification_history[-1]
        assert record["identified_model"] == "google/gemma-2-2b"
        assert record["confidence"] == 0.85
        assert record["metrics"] == metrics
        assert record["all_model_scores"] == all_scores
        assert "timestamp" in record
    
    def test_record_identification_history_limit(self, identifier):
        """Test that identification history is limited to prevent memory issues."""
        # Fill history beyond limit
        for i in range(1005):
            identifier._record_identification(
                "test_model", 0.8, {}, {}
            )
        
        # Should be limited to 1000 records
        assert len(identifier.identification_history) == 1000
    
    def test_add_model_signature(self, identifier):
        """Test adding new model signature."""
        initial_count = len(identifier.model_signatures)
        
        new_signature = ModelSignature(
            model_name="test/new-model",
            family=ModelFamily.GPT,
            typical_perplexity_range=(5.0, 15.0),
            token_distribution_entropy=6.0,
            repetition_patterns={"immediate": 0.01},
            sequence_coherence=8.0
        )
        
        identifier.add_model_signature(new_signature)
        
        assert len(identifier.model_signatures) == initial_count + 1
        assert "test/new-model" in identifier.model_signatures
        assert identifier.model_signatures["test/new-model"] == new_signature
    
    def test_get_identification_stats_empty(self, identifier):
        """Test getting identification statistics with empty history."""
        stats = identifier.get_identification_stats()
        
        assert stats["total_identifications"] == 0
    
    def test_get_identification_stats_with_data(self, identifier):
        """Test getting identification statistics with data."""
        # Add some test identifications
        identifier._record_identification("google/gemma-2-2b", 0.8, {}, {})
        identifier._record_identification("google/gemma-2-2b", 0.9, {}, {})
        identifier._record_identification("microsoft/Phi-3-mini-4k-instruct", 0.7, {}, {})
        
        stats = identifier.get_identification_stats()
        
        assert stats["total_identifications"] == 3
        assert stats["model_distribution"]["google/gemma-2-2b"] == 2
        assert stats["model_distribution"]["microsoft/Phi-3-mini-4k-instruct"] == 1
        assert abs(stats["average_confidence"] - 0.8) < 0.1  # (0.8 + 0.9 + 0.7) / 3
        assert stats["supported_models"] > 0
        assert stats["supported_families"] > 0
    
    def test_identify_model_error_handling(self, identifier):
        """Test error handling in model identification."""
        # Test with malformed metadata that causes exception
        malformed_metadata = {
            "detection_scores": "not_a_dict",  # Should be dict
            "text_analysis": None
        }
        
        model_name, confidence = identifier.identify_model(
            malformed_metadata, 0.8, 100
        )
        
        # Should handle error gracefully
        assert model_name is None
        assert confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__])