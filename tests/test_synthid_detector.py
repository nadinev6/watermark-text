"""
Unit tests for SynthID watermark detector.

This module tests the SynthIDDetector class functionality including
model loading, text analysis, and watermark detection capabilities.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import torch
from backend.detectors.synthid_detector import SynthIDDetector
from backend.detectors.base import DetectionResult, DetectionMethod, DetectionError


class TestSynthIDDetector:
    """Test cases for SynthIDDetector functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = MagicMock()
        config.model.gemma_model_name = "google/gemma-2-2b"
        config.model.model_cache_dir = "./test_cache"
        config.model.cpu_threads = 2
        config.model.max_model_memory_mb = 1024
        return config
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        tokenizer.eos_token = "[EOS]"
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        
        # Mock tokenization
        def mock_tokenize(text, **kwargs):
            # Simple mock: return word count as token count
            words = text.split()
            token_ids = list(range(len(words)))
            
            mock_encoding = MagicMock()
            mock_encoding.input_ids = [torch.tensor(token_ids)]
            return mock_encoding
        
        tokenizer.side_effect = mock_tokenize
        return tokenizer
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = MagicMock()
        model.eval.return_value = None
        
        # Mock model forward pass
        def mock_forward(input_ids, labels=None):
            mock_output = MagicMock()
            mock_output.loss = torch.tensor(2.5)  # Mock loss value
            return mock_output
        
        model.side_effect = mock_forward
        return model
    
    @patch('backend.detectors.synthid_detector.get_config')
    def test_detector_initialization(self, mock_get_config, mock_config):
        """Test SynthID detector initialization."""
        mock_get_config.return_value = mock_config
        
        detector = SynthIDDetector()
        
        assert detector.name == "SynthID Detector"
        assert detector.model_name == "google/gemma-2-2b"
        assert not detector._is_initialized
        assert detector.tokenizer is None
        assert detector.model is None
    
    @patch('backend.detectors.synthid_detector.get_config')
    def test_detector_initialization_custom_model(self, mock_get_config, mock_config):
        """Test detector initialization with custom model name."""
        mock_get_config.return_value = mock_config
        
        custom_model = "google/gemma-2-9b"
        detector = SynthIDDetector(model_name=custom_model)
        
        assert detector.model_name == custom_model
    
    @patch('backend.detectors.synthid_detector.AutoModelForCausalLM')
    @patch('backend.detectors.synthid_detector.AutoTokenizer')
    @patch('backend.detectors.synthid_detector.get_config')
    def test_lazy_model_loading(self, mock_get_config, mock_tokenizer_class, mock_model_class, mock_config):
        """Test lazy loading of model and tokenizer."""
        mock_get_config.return_value = mock_config
        
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        detector = SynthIDDetector()
        
        # Model should not be loaded initially
        assert not detector._is_initialized
        
        # Trigger lazy loading
        detector._lazy_load_model()
        
        # Verify model is loaded
        assert detector._is_initialized
        assert detector.tokenizer is not None
        assert detector.model is not None
        
        # Verify tokenizer configuration
        assert detector.tokenizer.pad_token == "[EOS]"  # Should be set to eos_token
    
    @patch('backend.detectors.synthid_detector.AutoTokenizer')
    @patch('backend.detectors.synthid_detector.get_config')
    def test_tokenizer_loading_failure(self, mock_get_config, mock_tokenizer_class, mock_config):
        """Test handling of tokenizer loading failure."""
        mock_get_config.return_value = mock_config
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Tokenizer load failed")
        
        detector = SynthIDDetector()
        
        with pytest.raises(DetectionError, match="Failed to load tokenizer"):
            detector._lazy_load_model()
    
    @patch('backend.detectors.synthid_detector.AutoModelForCausalLM')
    @patch('backend.detectors.synthid_detector.AutoTokenizer')
    @patch('backend.detectors.synthid_detector.get_config')
    def test_model_loading_failure(self, mock_get_config, mock_tokenizer_class, mock_model_class, mock_config):
        """Test handling of model loading failure."""
        mock_get_config.return_value = mock_config
        
        # Tokenizer loads successfully
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        
        # Model loading fails
        mock_model_class.from_pretrained.side_effect = Exception("Model load failed")
        
        detector = SynthIDDetector()
        
        with pytest.raises(DetectionError, match="Failed to load model"):
            detector._lazy_load_model()
    
    def test_detect_empty_text(self):
        """Test detection with empty text input."""
        detector = SynthIDDetector()
        
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            detector.detect("")
        
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            detector.detect("   ")
    
    def test_detect_short_text(self):
        """Test detection with text too short for reliable analysis."""
        detector = SynthIDDetector()
        
        with pytest.raises(ValueError, match="Text too short for reliable watermark detection"):
            detector.detect("Short")
    
    @patch('backend.detectors.synthid_detector.AutoModelForCausalLM')
    @patch('backend.detectors.synthid_detector.AutoTokenizer')
    @patch('backend.detectors.synthid_detector.get_config')
    def test_successful_detection(self, mock_get_config, mock_tokenizer_class, mock_model_class, mock_config):
        """Test successful watermark detection process."""
        mock_get_config.return_value = mock_config
        
        # Setup tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        
        # Mock tokenization
        mock_encoding = MagicMock()
        mock_encoding.input_ids = [torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Setup model mock
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(2.5)
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model
        
        detector = SynthIDDetector()
        
        # Test detection
        test_text = "This is a test text that is long enough for watermark detection analysis."
        result = detector.detect(test_text)
        
        # Verify result structure
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.is_watermarked, bool)
        assert result.detection_method == DetectionMethod.SYNTHID.value
        assert isinstance(result.metadata, dict)
        assert result.processing_time_ms >= 0
        
        # Verify metadata structure
        metadata = result.metadata
        assert "model_info" in metadata
        assert "text_analysis" in metadata
        assert "detection_scores" in metadata
        assert "detection_params" in metadata
        assert "performance" in metadata
    
    @patch('backend.detectors.synthid_detector.get_config')
    def test_tokenize_text(self, mock_get_config, mock_config):
        """Test text tokenization functionality."""
        mock_get_config.return_value = mock_config
        
        detector = SynthIDDetector()
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.input_ids = [torch.tensor([1, 2, 3, 4, 5])]
        mock_tokenizer.return_value = mock_encoding
        detector.tokenizer = mock_tokenizer
        
        # Test tokenization
        tokens = detector._tokenize_text("Test text for tokenization")
        
        assert tokens == [1, 2, 3, 4, 5]
        mock_tokenizer.assert_called_once()
    
    def test_tokenization_failure(self):
        """Test handling of tokenization failure."""
        detector = SynthIDDetector()
        
        # Mock failing tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = Exception("Tokenization failed")
        detector.tokenizer = mock_tokenizer
        
        with pytest.raises(DetectionError, match="Text tokenization failed"):
            detector._tokenize_text("Test text")
    
    @patch('backend.detectors.synthid_detector.get_config')
    def test_perplexity_calculation(self, mock_get_config, mock_config):
        """Test perplexity-based detection score calculation."""
        mock_get_config.return_value = mock_config
        
        detector = SynthIDDetector()
        
        # Mock model
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(2.0)  # Loss value
        mock_model.return_value = mock_output
        detector.model = mock_model
        
        # Test perplexity calculation
        tokens = [1, 2, 3, 4, 5]
        score = detector._calculate_perplexity_score(tokens)
        
        assert 0.0 <= score <= 1.0
        mock_model.assert_called_once()
    
    def test_perplexity_calculation_failure(self):
        """Test handling of perplexity calculation failure."""
        detector = SynthIDDetector()
        
        # Mock failing model
        mock_model = MagicMock()
        mock_model.side_effect = Exception("Model inference failed")
        detector.model = mock_model
        
        # Should return neutral score on failure
        tokens = [1, 2, 3, 4, 5]
        score = detector._calculate_perplexity_score(tokens)
        
        assert score == 0.5  # Neutral score
    
    def test_token_distribution_analysis(self):
        """Test token distribution analysis."""
        detector = SynthIDDetector()
        
        # Test with varied token distribution
        tokens = [1, 2, 3, 1, 2, 4, 5, 1]  # Some repetition
        score = detector._analyze_token_distribution(tokens)
        
        assert 0.0 <= score <= 1.0
    
    def test_repetition_pattern_analysis(self):
        """Test repetition pattern analysis."""
        detector = SynthIDDetector()
        
        # Test with repetitive tokens
        tokens = [1, 1, 2, 2, 3, 3, 4, 4]  # High repetition
        score = detector._analyze_repetition_patterns(tokens)
        
        assert 0.0 <= score <= 1.0
        
        # Test with non-repetitive tokens
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # Low repetition
        score2 = detector._analyze_repetition_patterns(tokens)
        
        assert 0.0 <= score2 <= 1.0
        assert score > score2  # More repetitive should have higher score
    
    def test_sequence_coherence_analysis(self):
        """Test sequence coherence analysis."""
        detector = SynthIDDetector()
        
        # Test with token sequence
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        score = detector._analyze_sequence_coherence(tokens)
        
        assert 0.0 <= score <= 1.0
        
        # Test with single token
        single_token = [1]
        score_single = detector._analyze_sequence_coherence(single_token)
        
        assert score_single == 0.5  # Should return neutral score
    
    def test_confidence_score_calculation(self):
        """Test overall confidence score calculation."""
        detector = SynthIDDetector()
        
        # Test with sample detection scores
        detection_scores = {
            "perplexity": 0.8,
            "token_distribution": 0.6,
            "repetition_patterns": 0.7,
            "sequence_coherence": 0.5
        }
        
        confidence = detector._calculate_confidence_score(detection_scores)
        
        assert 0.0 <= confidence <= 1.0
        
        # Should be weighted average: 0.8*0.4 + 0.6*0.3 + 0.7*0.2 + 0.5*0.1 = 0.69
        expected = 0.69
        assert abs(confidence - expected) < 0.01
    
    def test_confidence_score_missing_metrics(self):
        """Test confidence calculation with missing metrics."""
        detector = SynthIDDetector()
        
        # Test with partial scores
        detection_scores = {
            "perplexity": 0.8,
            "unknown_metric": 0.9  # Unknown metric should use default weight
        }
        
        confidence = detector._calculate_confidence_score(detection_scores)
        
        assert 0.0 <= confidence <= 1.0
    
    @patch('backend.detectors.synthid_detector.AutoModelForCausalLM')
    @patch('backend.detectors.synthid_detector.AutoTokenizer')
    @patch('backend.detectors.synthid_detector.get_config')
    def test_is_available(self, mock_get_config, mock_tokenizer_class, mock_model_class, mock_config):
        """Test detector availability check."""
        mock_get_config.return_value = mock_config
        
        detector = SynthIDDetector()
        
        # Should not be available initially
        assert not detector.is_available()
        
        # Setup mocks for successful loading
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model_class.from_pretrained.return_value = MagicMock()
        
        # Should be available after loading
        assert detector.is_available()
        assert detector._is_initialized
    
    def test_is_available_loading_failure(self):
        """Test availability check when loading fails."""
        detector = SynthIDDetector()
        
        # Mock failed loading
        with patch.object(detector, '_lazy_load_model', side_effect=Exception("Load failed")):
            assert not detector.is_available()
    
    def test_get_supported_models(self):
        """Test getting list of supported models."""
        detector = SynthIDDetector()
        
        supported = detector.get_supported_models()
        
        assert isinstance(supported, list)
        assert len(supported) > 0
        assert "google/gemma-2-2b" in supported
    
    def test_cleanup(self):
        """Test resource cleanup."""
        detector = SynthIDDetector()
        
        # Set up some mock resources
        detector.model = MagicMock()
        detector.tokenizer = MagicMock()
        detector._is_initialized = True
        
        # Cleanup
        detector.cleanup()
        
        assert detector.model is None
        assert detector.tokenizer is None
        assert not detector._is_initialized
    
    def test_get_model_info(self):
        """Test getting model information."""
        detector = SynthIDDetector()
        
        info = detector.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "is_loaded" in info
        assert "supported_models" in info
        assert "config" in info
        
        assert info["model_name"] == detector.model_name
        assert info["is_loaded"] == detector._is_initialized


if __name__ == "__main__":
    pytest.main([__file__])