"""
Unit tests for test data generation pipeline.

This module tests the TestDataGenerator class functionality including
watermarked and clean sample generation, dataset creation, and validation.
"""

import pytest
import tempfile
import json
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import torch

from backend.utils.test_data_generator import (
    TestDataGenerator,
    GenerationParams,
    create_test_generator,
    generate_sample_dataset,
    create_evaluation_prompts
)
from backend.models.schemas import TestSample


class TestGenerationParams:
    """Test cases for GenerationParams dataclass."""
    
    def test_default_params(self):
        """Test default generation parameters."""
        params = GenerationParams()
        
        assert params.max_new_tokens == 150
        assert params.temperature == 0.8
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.do_sample is True
        assert params.repetition_penalty == 1.1
        assert params.length_penalty == 1.0
        assert params.no_repeat_ngram_size == 3
    
    def test_custom_params(self):
        """Test custom generation parameters."""
        params = GenerationParams(
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=False
        )
        
        assert params.max_new_tokens == 200
        assert params.temperature == 0.7
        assert params.top_p == 0.95
        assert params.do_sample is False


class TestTestDataGenerator:
    """Test cases for TestDataGenerator functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = MagicMock()
        config.model.gemma_model_name = "google/gemma-2-2b"
        config.model.phi_model_name = "microsoft/Phi-3-mini-4k-instruct"
        config.model.model_cache_dir = "./test_cache"
        config.model.cpu_threads = 2
        return config
    
    @pytest.fixture
    def sample_prompts(self):
        """Sample prompts for testing."""
        return [
            "Write about artificial intelligence.",
            "Describe the benefits of renewable energy.",
            "Explain the importance of education."
        ]
    
    @patch('backend.utils.test_data_generator.get_config')
    def test_generator_initialization(self, mock_get_config, mock_config):
        """Test TestDataGenerator initialization."""
        mock_get_config.return_value = mock_config
        
        generator = TestDataGenerator()
        
        assert generator.gemma_model_name == "google/gemma-2-2b"
        assert generator.phi_model_name == "microsoft/Phi-3-mini-4k-instruct"
        assert generator.gemma_model is None
        assert generator.phi_model is None
        assert generator.generation_count == 0
        assert len(generator.default_prompts) > 0
    
    @patch('backend.utils.test_data_generator.AutoModelForCausalLM')
    @patch('backend.utils.test_data_generator.AutoTokenizer')
    @patch('backend.utils.test_data_generator.get_config')
    def test_load_gemma_model(self, mock_get_config, mock_tokenizer_class, mock_model_class, mock_config):
        """Test Gemma model loading."""
        mock_get_config.return_value = mock_config
        
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        generator = TestDataGenerator()
        generator._load_gemma_model()
        
        # Verify model loading
        assert generator.gemma_tokenizer is not None
        assert generator.gemma_model is not None
        assert generator.gemma_tokenizer.pad_token == "[EOS]"
        
        # Verify model configuration calls
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
    
    @patch('backend.utils.test_data_generator.AutoTokenizer')
    @patch('backend.utils.test_data_generator.get_config')
    def test_load_gemma_model_failure(self, mock_get_config, mock_tokenizer_class, mock_config):
        """Test handling of Gemma model loading failure."""
        mock_get_config.return_value = mock_config
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Model load failed")
        
        generator = TestDataGenerator()
        
        with pytest.raises(RuntimeError, match="Gemma model loading failed"):
            generator._load_gemma_model()
    
    @patch('backend.utils.test_data_generator.AutoModelForCausalLM')
    @patch('backend.utils.test_data_generator.AutoTokenizer')
    @patch('backend.utils.test_data_generator.get_config')
    def test_load_phi_model(self, mock_get_config, mock_tokenizer_class, mock_model_class, mock_config):
        """Test Phi model loading."""
        mock_get_config.return_value = mock_config
        
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        generator = TestDataGenerator()
        generator._load_phi_model()
        
        # Verify model loading
        assert generator.phi_tokenizer is not None
        assert generator.phi_model is not None
        assert generator.phi_tokenizer.pad_token == "[EOS]"
    
    def test_generate_watermarked_samples_invalid_count(self):
        """Test watermarked sample generation with invalid count."""
        generator = TestDataGenerator()
        
        with pytest.raises(ValueError, match="Sample count must be positive"):
            generator.generate_watermarked_samples(0)
        
        with pytest.raises(ValueError, match="Sample count must be positive"):
            generator.generate_watermarked_samples(-1)
    
    def test_generate_watermarked_samples_empty_prompts(self):
        """Test watermarked sample generation with empty prompts."""
        generator = TestDataGenerator()
        
        with pytest.raises(ValueError, match="Prompts list cannot be empty"):
            generator.generate_watermarked_samples(5, prompts=[])
    
    @patch('backend.utils.test_data_generator.AutoModelForCausalLM')
    @patch('backend.utils.test_data_generator.AutoTokenizer')
    @patch('backend.utils.test_data_generator.get_config')
    def test_generate_watermarked_samples_success(self, mock_get_config, mock_tokenizer_class, mock_model_class, mock_config, sample_prompts):
        """Test successful watermarked sample generation."""
        mock_get_config.return_value = mock_config
        
        # Setup tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.eos_token = "[EOS]"
        
        # Mock tokenization
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = mock_inputs
        
        # Mock decoding
        mock_tokenizer.decode.return_value = "This is a generated watermarked text sample."
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Setup model mock
        mock_model = MagicMock()
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # Mock generated tokens
        mock_model.generate.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        generator = TestDataGenerator()
        samples = generator.generate_watermarked_samples(2, prompts=sample_prompts)
        
        # Verify results
        assert len(samples) == 2
        for sample in samples:
            assert isinstance(sample, TestSample)
            assert sample.is_watermarked is True
            assert 0.7 <= sample.expected_score <= 0.95
            assert sample.source_model == "google/gemma-2-2b"
            assert len(sample.text) > 0
    
    def test_generate_clean_samples_invalid_count(self):
        """Test clean sample generation with invalid count."""
        generator = TestDataGenerator()
        
        with pytest.raises(ValueError, match="Sample count must be positive"):
            generator.generate_clean_samples(0)
    
    @patch('backend.utils.test_data_generator.AutoModelForCausalLM')
    @patch('backend.utils.test_data_generator.AutoTokenizer')
    @patch('backend.utils.test_data_generator.get_config')
    def test_generate_clean_samples_success(self, mock_get_config, mock_tokenizer_class, mock_model_class, mock_config, sample_prompts):
        """Test successful clean sample generation."""
        mock_get_config.return_value = mock_config
        
        # Setup tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.eos_token = "[EOS]"
        
        # Mock tokenization
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = mock_inputs
        
        # Mock decoding
        mock_tokenizer.decode.return_value = "This is a generated clean text sample."
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Setup model mock
        mock_model = MagicMock()
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        mock_model.generate.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        generator = TestDataGenerator()
        samples = generator.generate_clean_samples(2, prompts=sample_prompts)
        
        # Verify results
        assert len(samples) == 2
        for sample in samples:
            assert isinstance(sample, TestSample)
            assert sample.is_watermarked is False
            assert 0.05 <= sample.expected_score <= 0.3
            assert sample.source_model == "microsoft/Phi-3-mini-4k-instruct"
            assert len(sample.text) > 0
    
    def test_add_prompt_variation(self):
        """Test prompt variation generation."""
        generator = TestDataGenerator()
        
        original_prompt = "Write about artificial intelligence."
        
        # Test different variations
        variation_0 = generator._add_prompt_variation(original_prompt, 0)
        variation_1 = generator._add_prompt_variation(original_prompt, 1)
        variation_3 = generator._add_prompt_variation(original_prompt, 3)
        
        # Should get different variations
        assert variation_0.startswith("Please")
        assert variation_1.startswith("Could you")
        assert variation_3 == original_prompt  # Index 3 should return original
    
    def test_clean_generated_text(self):
        """Test text cleaning functionality."""
        generator = TestDataGenerator()
        
        # Test with extra whitespace
        messy_text = "  This   is   a   test   text.  "
        cleaned = generator._clean_generated_text(messy_text)
        assert cleaned == "This is a test text."
        
        # Test with incomplete sentence
        incomplete_text = "This is a complete sentence. This is incomp"
        cleaned = generator._clean_generated_text(incomplete_text)
        assert cleaned == "This is a complete sentence."
        
        # Test with short text (should be extended)
        short_text = "Short."
        cleaned = generator._clean_generated_text(short_text)
        assert len(cleaned.split()) >= 20
    
    @patch.object(TestDataGenerator, 'generate_watermarked_samples')
    @patch.object(TestDataGenerator, 'generate_clean_samples')
    def test_create_balanced_dataset(self, mock_clean_gen, mock_watermarked_gen):
        """Test balanced dataset creation."""
        # Setup mocks
        mock_watermarked_samples = [
            TestSample(text="Watermarked 1", is_watermarked=True, expected_score=0.8, source_model="gemma", generation_params={}),
            TestSample(text="Watermarked 2", is_watermarked=True, expected_score=0.9, source_model="gemma", generation_params={})
        ]
        mock_clean_samples = [
            TestSample(text="Clean 1", is_watermarked=False, expected_score=0.2, source_model="phi", generation_params={}),
            TestSample(text="Clean 2", is_watermarked=False, expected_score=0.1, source_model="phi", generation_params={})
        ]
        
        mock_watermarked_gen.return_value = mock_watermarked_samples
        mock_clean_gen.return_value = mock_clean_samples
        
        generator = TestDataGenerator()
        dataset = generator.create_balanced_dataset(total_size=4, watermarked_ratio=0.5)
        
        # Verify dataset
        assert len(dataset) == 4
        watermarked_count = sum(1 for s in dataset if s.is_watermarked)
        clean_count = sum(1 for s in dataset if not s.is_watermarked)
        
        assert watermarked_count == 2
        assert clean_count == 2
    
    def test_create_balanced_dataset_invalid_params(self):
        """Test balanced dataset creation with invalid parameters."""
        generator = TestDataGenerator()
        
        with pytest.raises(ValueError, match="Total size must be positive"):
            generator.create_balanced_dataset(0)
        
        with pytest.raises(ValueError, match="Watermarked ratio must be between 0.0 and 1.0"):
            generator.create_balanced_dataset(10, watermarked_ratio=1.5)
    
    def test_save_and_load_dataset(self):
        """Test dataset saving and loading."""
        generator = TestDataGenerator()
        
        # Create sample dataset
        samples = [
            TestSample(text="Sample 1", is_watermarked=True, expected_score=0.8, source_model="gemma", generation_params={}),
            TestSample(text="Sample 2", is_watermarked=False, expected_score=0.2, source_model="phi", generation_params={})
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save dataset
            generator.save_dataset(samples, temp_path)
            
            # Verify file exists
            assert Path(temp_path).exists()
            
            # Load dataset
            loaded_samples, metadata = generator.load_dataset(temp_path)
            
            # Verify loaded data
            assert len(loaded_samples) == 2
            assert loaded_samples[0].text == "Sample 1"
            assert loaded_samples[0].is_watermarked is True
            assert loaded_samples[1].text == "Sample 2"
            assert loaded_samples[1].is_watermarked is False
            
            # Verify metadata
            assert "total_samples" in metadata
            assert metadata["total_samples"] == 2
            
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validate_dataset(self):
        """Test dataset validation."""
        generator = TestDataGenerator()
        
        # Create valid dataset
        valid_samples = [
            TestSample(text="Valid watermarked text", is_watermarked=True, expected_score=0.8, source_model="gemma", generation_params={}),
            TestSample(text="Valid clean text", is_watermarked=False, expected_score=0.2, source_model="phi", generation_params={})
        ]
        
        validation_results = generator.validate_dataset(valid_samples)
        
        assert validation_results["valid"] is True
        assert validation_results["total_samples"] == 2
        assert validation_results["watermarked_count"] == 1
        assert validation_results["clean_count"] == 1
        assert len(validation_results["validation_errors"]) == 0
    
    def test_validate_dataset_with_errors(self):
        """Test dataset validation with errors."""
        generator = TestDataGenerator()
        
        # Create invalid dataset
        invalid_samples = [
            TestSample(text="", is_watermarked=True, expected_score=0.8, source_model="gemma", generation_params={}),  # Empty text
            TestSample(text="Valid text", is_watermarked=True, expected_score=0.2, source_model="gemma", generation_params={}),  # Low score for watermarked
            TestSample(text="Valid text", is_watermarked=False, expected_score=1.5, source_model="phi", generation_params={})  # Invalid score
        ]
        
        validation_results = generator.validate_dataset(invalid_samples)
        
        assert validation_results["valid"] is False
        assert len(validation_results["validation_errors"]) > 0
    
    def test_validate_empty_dataset(self):
        """Test validation of empty dataset."""
        generator = TestDataGenerator()
        
        validation_results = generator.validate_dataset([])
        
        assert validation_results["valid"] is False
        assert validation_results["error"] == "Empty dataset"
    
    def test_get_generation_stats(self):
        """Test generation statistics."""
        generator = TestDataGenerator()
        
        stats = generator.get_generation_stats()
        
        assert "total_generations" in stats
        assert "total_tokens_generated" in stats
        assert "gemma_model_loaded" in stats
        assert "phi_model_loaded" in stats
        assert "gemma_model_name" in stats
        assert "phi_model_name" in stats
        assert "default_prompts_count" in stats
        
        assert stats["total_generations"] == 0  # Initially zero
        assert stats["gemma_model_loaded"] is False  # Not loaded initially
        assert stats["phi_model_loaded"] is False  # Not loaded initially
    
    def test_cleanup(self):
        """Test resource cleanup."""
        generator = TestDataGenerator()
        
        # Set up some mock resources
        generator.gemma_model = MagicMock()
        generator.gemma_tokenizer = MagicMock()
        generator.phi_model = MagicMock()
        generator.phi_tokenizer = MagicMock()
        
        # Cleanup
        generator.cleanup()
        
        assert generator.gemma_model is None
        assert generator.gemma_tokenizer is None
        assert generator.phi_model is None
        assert generator.phi_tokenizer is None


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    @patch('backend.utils.test_data_generator.get_config')
    def test_create_test_generator(self, mock_get_config):
        """Test creating test generator with factory function."""
        mock_config = MagicMock()
        mock_config.model.gemma_model_name = "google/gemma-2-2b"
        mock_config.model.phi_model_name = "microsoft/Phi-3-mini-4k-instruct"
        mock_get_config.return_value = mock_config
        
        generator = create_test_generator()
        
        assert isinstance(generator, TestDataGenerator)
        assert generator.gemma_model_name == "google/gemma-2-2b"
        assert generator.phi_model_name == "microsoft/Phi-3-mini-4k-instruct"
    
    @patch.object(TestDataGenerator, 'create_balanced_dataset')
    @patch.object(TestDataGenerator, 'validate_dataset')
    @patch.object(TestDataGenerator, 'save_dataset')
    @patch.object(TestDataGenerator, 'cleanup')
    @patch('backend.utils.test_data_generator.create_test_generator')
    def test_generate_sample_dataset(self, mock_create_gen, mock_cleanup, mock_save, mock_validate, mock_create_balanced):
        """Test sample dataset generation function."""
        # Setup mocks
        mock_generator = MagicMock()
        mock_samples = [TestSample(text="Test", is_watermarked=True, expected_score=0.8, source_model="test", generation_params={})]
        mock_validation = {"valid": True, "total_samples": 1}
        
        mock_create_gen.return_value = mock_generator
        mock_generator.create_balanced_dataset.return_value = mock_samples
        mock_generator.validate_dataset.return_value = mock_validation
        
        # Test function
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            generate_sample_dataset(temp_path, total_samples=10, watermarked_ratio=0.6)
            
            # Verify calls
            mock_generator.create_balanced_dataset.assert_called_once_with(
                total_size=10,
                watermarked_ratio=0.6,
                prompts=None
            )
            mock_generator.validate_dataset.assert_called_once_with(mock_samples)
            mock_generator.save_dataset.assert_called_once_with(mock_samples, temp_path)
            mock_generator.cleanup.assert_called_once()
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_create_evaluation_prompts(self):
        """Test evaluation prompts creation."""
        prompts = create_evaluation_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        
        # Verify all prompts are strings
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 0


if __name__ == "__main__":
    pytest.main([__file__])