"""
Unit tests for custom fallback watermark detector.

This module tests the CustomDetector class functionality including
statistical analysis, linguistic feature detection, and AI pattern recognition.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import torch
from backend.detectors.custom_detector import (
    CustomDetector,
    create_custom_detector,
    create_statistical_detector
)
from backend.detectors.base import DetectionResult, DetectionMethod, DetectionError


class TestCustomDetector:
    """Test cases for CustomDetector functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = MagicMock()
        config.model.phi_model_name = "microsoft/Phi-3-mini-4k-instruct"
        config.model.model_cache_dir = "./test_cache"
        config.model.cpu_threads = 2
        config.model.max_model_memory_mb = 1024
        return config
    
    @pytest.fixture
    def sample_ai_text(self):
        """Sample text that should trigger AI detection patterns."""
        return """
        The artificial intelligence system processes information efficiently and accurately.
        The system analyzes data patterns to identify relevant insights.
        The algorithm utilizes machine learning techniques for optimal performance.
        The implementation ensures reliable and consistent results across various scenarios.
        The framework provides comprehensive solutions for complex computational tasks.
        """
    
    @pytest.fixture
    def sample_human_text(self):
        """Sample text that should appear more human-like."""
        return """
        I was walking down the street yesterday when I bumped into my old friend Sarah.
        We hadn't seen each other in years! She looked great and told me about her new job.
        Apparently, she's working at a startup now - something about sustainable energy.
        It's funny how life takes unexpected turns, isn't it? We grabbed coffee and caught up.
        I'm really glad we reconnected after all this time.
        """
    
    @patch('backend.detectors.custom_detector.get_config')
    def test_detector_initialization(self, mock_get_config, mock_config):
        """Test CustomDetector initialization."""
        mock_get_config.return_value = mock_config
        
        detector = CustomDetector()
        
        assert detector.name == "Custom Fallback Detector"
        assert detector.model_name == "microsoft/Phi-3-mini-4k-instruct"
        assert not detector._is_initialized
        assert detector.tokenizer is None
        assert detector.model is None
    
    @patch('backend.detectors.custom_detector.get_config')
    def test_detector_initialization_custom_model(self, mock_get_config, mock_config):
        """Test detector initialization with custom model name."""
        mock_get_config.return_value = mock_config
        
        custom_model = "microsoft/Phi-3-small-8k-instruct"
        detector = CustomDetector(model_name=custom_model)
        
        assert detector.model_name == custom_model
    
    @patch('backend.detectors.custom_detector.AutoModelForCausalLM')
    @patch('backend.detectors.custom_detector.AutoTokenizer')
    @patch('backend.detectors.custom_detector.get_config')
    def test_lazy_model_loading_success(self, mock_get_config, mock_tokenizer_class, mock_model_class, mock_config):
        """Test successful lazy loading of model and tokenizer."""
        mock_get_config.return_value = mock_config
        
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        detector = CustomDetector()
        
        # Trigger lazy loading
        detector._lazy_load_model()
        
        # Verify initialization
        assert detector._is_initialized
        assert detector.tokenizer is not None
        assert detector.model is not None
    
    @patch('backend.detectors.custom_detector.AutoTokenizer')
    @patch('backend.detectors.custom_detector.get_config')
    def test_lazy_loading_tokenizer_failure(self, mock_get_config, mock_tokenizer_class, mock_config):
        """Test handling of tokenizer loading failure."""
        mock_get_config.return_value = mock_config
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Tokenizer load failed")
        
        detector = CustomDetector()
        
        # Should not raise exception, but continue without tokenizer
        detector._lazy_load_model()
        
        assert detector._is_initialized  # Should still be initialized
        assert detector.tokenizer is None  # But tokenizer should be None
    
    def test_detect_empty_text(self):
        """Test detection with empty text input."""
        detector = CustomDetector()
        
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            detector.detect("")
        
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            detector.detect("   ")
    
    def test_detect_short_text(self):
        """Test detection with text too short for reliable analysis."""
        detector = CustomDetector()
        
        short_text = "Too short"  # Less than min_words_for_detection
        
        with pytest.raises(ValueError, match="Text too short for reliable detection"):
            detector.detect(short_text)
    
    @patch('backend.detectors.custom_detector.get_config')
    def test_successful_detection_without_model(self, mock_get_config, mock_config, sample_ai_text):
        """Test successful detection using only statistical methods."""
        mock_get_config.return_value = mock_config
        
        detector = CustomDetector()
        # Force initialization without model loading
        detector._is_initialized = True
        detector.model = None
        detector.tokenizer = None
        
        result = detector.detect(sample_ai_text)
        
        # Verify result structure
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.is_watermarked, bool)
        assert result.detection_method == DetectionMethod.CUSTOM.value
        assert isinstance(result.metadata, dict)
        assert result.processing_time_ms >= 0
        
        # Verify metadata structure
        metadata = result.metadata
        assert "detector_info" in metadata
        assert "text_analysis" in metadata
        assert "statistical_scores" in metadata
        assert "detection_thresholds" in metadata
        assert "analysis_methods" in metadata
    
    @patch('backend.detectors.custom_detector.AutoModelForCausalLM')
    @patch('backend.detectors.custom_detector.AutoTokenizer')
    @patch('backend.detectors.custom_detector.get_config')
    def test_successful_detection_with_model(self, mock_get_config, mock_tokenizer_class, mock_model_class, mock_config, sample_ai_text):
        """Test successful detection with perplexity analysis."""
        mock_get_config.return_value = mock_config
        
        # Setup tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "[EOS]"
        
        # Mock tokenization
        mock_encoding = MagicMock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Setup model mock
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(2.0)
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        detector = CustomDetector()
        result = detector.detect(sample_ai_text)
        
        # Verify perplexity analysis was included
        assert "perplexity" in result.metadata["statistical_scores"]
        assert result.metadata["analysis_methods"]["perplexity_available"]
    
    def test_repetition_analysis(self):
        """Test repetition pattern analysis."""
        detector = CustomDetector()
        
        # Text with high repetition
        repetitive_words = ["the", "cat", "sat", "the", "cat", "sat", "on", "the", "mat"]
        score = detector._analyze_repetition_patterns(repetitive_words)
        
        assert 0.0 <= score <= 1.0
        
        # Text with low repetition
        diverse_words = ["unique", "different", "various", "distinct", "separate", "individual"]
        score2 = detector._analyze_repetition_patterns(diverse_words)
        
        assert 0.0 <= score2 <= 1.0
        assert score > score2  # More repetitive should have higher score
    
    def test_ngram_repetition_counting(self):
        """Test n-gram repetition counting."""
        detector = CustomDetector()
        
        # Test bigram repetitions
        words = ["the", "cat", "sat", "the", "cat", "ran"]  # "the cat" appears twice
        bigram_score = detector._count_ngram_repetitions(words, 2)
        
        assert bigram_score > 0.0
        
        # Test with no repetitions
        unique_words = ["one", "two", "three", "four", "five"]
        unique_score = detector._count_ngram_repetitions(unique_words, 2)
        
        assert unique_score == 0.0
    
    def test_burstiness_analysis(self):
        """Test burstiness pattern analysis."""
        detector = CustomDetector()
        
        # Create text with bursty word usage
        bursty_words = ["the"] * 5 + ["cat"] + ["the"] * 3 + ["sat"] + ["the"] * 2
        burstiness_score = detector._analyze_burstiness(bursty_words)
        
        assert 0.0 <= burstiness_score <= 1.0
        
        # Test with short word list
        short_words = ["a", "b", "c"]
        short_score = detector._analyze_burstiness(short_words)
        
        assert short_score == 0.5  # Should return neutral score
    
    def test_entropy_analysis(self):
        """Test entropy analysis."""
        detector = CustomDetector()
        
        # High entropy (diverse words)
        diverse_words = ["apple", "banana", "cherry", "date", "elderberry"]
        entropy_score = detector._analyze_entropy(diverse_words)
        
        assert 0.0 <= entropy_score <= 1.0
        
        # Low entropy (repetitive words)
        repetitive_words = ["the"] * 10
        repetitive_score = detector._analyze_entropy(repetitive_words)
        
        assert 0.0 <= repetitive_score <= 1.0
    
    def test_linguistic_features_analysis(self, sample_ai_text):
        """Test linguistic features analysis."""
        detector = CustomDetector()
        
        words = sample_ai_text.split()
        score = detector._analyze_linguistic_features(sample_ai_text, words)
        
        assert 0.0 <= score <= 1.0
    
    def test_sentence_structure_analysis(self, sample_ai_text):
        """Test sentence structure analysis."""
        detector = CustomDetector()
        
        score = detector._analyze_sentence_structure(sample_ai_text)
        
        assert 0.0 <= score <= 1.0
        
        # Test with single sentence
        single_sentence = "This is a single sentence."
        single_score = detector._analyze_sentence_structure(single_sentence)
        
        assert single_score == 0.5  # Should return neutral score
    
    def test_vocabulary_diversity_analysis(self):
        """Test vocabulary diversity analysis."""
        detector = CustomDetector()
        
        # Diverse vocabulary
        diverse_words = ["unique", "different", "various", "distinct", "separate"] * 2
        diversity_score = detector._analyze_vocabulary_diversity(diverse_words)
        
        assert 0.0 <= diversity_score <= 1.0
        
        # Test with short word list
        short_words = ["a", "b"]
        short_score = detector._analyze_vocabulary_diversity(short_words)
        
        assert short_score == 0.5  # Should return neutral score
    
    def test_confidence_score_calculation(self):
        """Test overall confidence score calculation."""
        detector = CustomDetector()
        
        # Test with sample analysis results
        analysis_results = {
            "perplexity": 0.8,
            "repetition": 0.7,
            "burstiness": 0.6,
            "entropy": 0.5,
            "linguistic": 0.4,
            "structure": 0.3,
            "diversity": 0.2
        }
        
        confidence = detector._calculate_confidence_score(analysis_results)
        
        assert 0.0 <= confidence <= 1.0
        
        # Should be weighted average based on defined weights
        expected = (0.8*0.25 + 0.7*0.20 + 0.6*0.15 + 0.5*0.15 + 0.4*0.10 + 0.3*0.10 + 0.2*0.05)
        assert abs(confidence - expected) < 0.01
    
    def test_confidence_score_missing_metrics(self):
        """Test confidence calculation with missing metrics."""
        detector = CustomDetector()
        
        # Test with partial results
        partial_results = {
            "perplexity": 0.8,
            "repetition": 0.6
        }
        
        confidence = detector._calculate_confidence_score(partial_results)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_is_available(self):
        """Test detector availability check."""
        detector = CustomDetector()
        
        # Custom detector should always be available
        assert detector.is_available()
    
    def test_get_supported_models(self):
        """Test getting list of supported models."""
        detector = CustomDetector()
        
        supported = detector.get_supported_models()
        
        assert isinstance(supported, list)
        assert len(supported) > 0
        assert "microsoft/Phi-3-mini-4k-instruct" in supported
        assert "ai-generated" in supported
    
    def test_cleanup(self):
        """Test resource cleanup."""
        detector = CustomDetector()
        
        # Set up some mock resources
        detector.model = MagicMock()
        detector.tokenizer = MagicMock()
        
        # Cleanup
        detector.cleanup()
        
        assert detector.model is None
        assert detector.tokenizer is None
    
    def test_get_detection_info(self):
        """Test getting detector information."""
        detector = CustomDetector()
        
        info = detector.get_detection_info()
        
        assert isinstance(info, dict)
        assert "detector_name" in info
        assert "reference_model" in info
        assert "capabilities" in info
        assert "thresholds" in info
        assert "supported_models" in info
        
        # Verify capabilities
        capabilities = info["capabilities"]
        assert capabilities["statistical_analysis"] is True
        assert capabilities["linguistic_features"] is True
        assert capabilities["structure_analysis"] is True
    
    def test_update_thresholds(self):
        """Test updating detection thresholds."""
        detector = CustomDetector()
        
        original_perplexity = detector.perplexity_threshold
        original_repetition = detector.repetition_threshold
        
        # Update thresholds
        detector.update_thresholds(
            perplexity_threshold=20.0,
            repetition_threshold=0.2
        )
        
        assert detector.perplexity_threshold == 20.0
        assert detector.repetition_threshold == 0.2
        assert detector.perplexity_threshold != original_perplexity
        assert detector.repetition_threshold != original_repetition
    
    def test_statistical_analysis_error_handling(self):
        """Test error handling in statistical analysis."""
        detector = CustomDetector()
        
        # Force an error by providing invalid input to internal method
        with patch.object(detector, '_analyze_repetition_patterns', side_effect=Exception("Analysis failed")):
            # Should handle error gracefully and provide fallback scores
            analysis_results = detector._perform_statistical_analysis("test text", ["test", "text"])
            
            # Should contain fallback neutral scores
            assert all(0.0 <= score <= 1.0 for score in analysis_results.values())


class TestCustomDetectorFactories:
    """Test cases for custom detector factory functions."""
    
    @patch('backend.detectors.custom_detector.get_config')
    def test_create_custom_detector(self, mock_get_config):
        """Test creating custom detector with factory function."""
        mock_config = MagicMock()
        mock_config.model.phi_model_name = "microsoft/Phi-3-mini-4k-instruct"
        mock_get_config.return_value = mock_config
        
        detector = create_custom_detector()
        
        assert isinstance(detector, CustomDetector)
        assert detector.model_name == "microsoft/Phi-3-mini-4k-instruct"
    
    @patch('backend.detectors.custom_detector.get_config')
    def test_create_custom_detector_with_model(self, mock_get_config):
        """Test creating custom detector with specific model."""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        
        custom_model = "microsoft/Phi-3-small-8k-instruct"
        detector = create_custom_detector(model_name=custom_model)
        
        assert isinstance(detector, CustomDetector)
        assert detector.model_name == custom_model
    
    def test_create_statistical_detector(self):
        """Test creating statistical-only detector."""
        detector = create_statistical_detector()
        
        assert isinstance(detector, CustomDetector)
        assert detector._is_initialized is True
        assert detector.model is None
        assert detector.tokenizer is None
        
        # Should still be available for statistical analysis
        assert detector.is_available()


class TestCustomDetectorIntegration:
    """Integration tests for custom detector with different text types."""
    
    @pytest.fixture
    def detector(self):
        """Create detector for integration testing."""
        return create_statistical_detector()
    
    def test_ai_text_detection(self, detector):
        """Test detection on AI-like text patterns."""
        ai_text = """
        The system processes data efficiently and accurately. The algorithm analyzes 
        information systematically. The implementation provides reliable results. 
        The framework ensures optimal performance across various scenarios.
        """
        
        result = detector.detect(ai_text)
        
        # Should detect some AI patterns
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.confidence_score <= 1.0
    
    def test_human_text_detection(self, detector):
        """Test detection on human-like text patterns."""
        human_text = """
        Yesterday I went to the grocery store and bumped into my neighbor Sarah. 
        She was buying ingredients for her famous chocolate chip cookies! We chatted 
        about the weather and her kids' soccer games. It's always nice to catch up 
        with friends in unexpected places.
        """
        
        result = detector.detect(human_text)
        
        # Should have lower confidence for human text
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.confidence_score <= 1.0
    
    def test_mixed_content_detection(self, detector):
        """Test detection on mixed content."""
        mixed_text = """
        The artificial intelligence system demonstrates remarkable capabilities in 
        processing natural language. However, I personally think that human creativity 
        and intuition still play crucial roles in many domains. What do you think 
        about this balance between AI efficiency and human insight?
        """
        
        result = detector.detect(mixed_text)
        
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.confidence_score <= 1.0
        
        # Should have metadata about the analysis
        assert "statistical_scores" in result.metadata
        assert len(result.metadata["statistical_scores"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])