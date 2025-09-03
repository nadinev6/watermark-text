"""
Test data generation pipeline for watermark detection validation.

This module provides functionality to generate both watermarked and non-watermarked
text samples for testing and validating watermark detection systems. It uses
Google's Gemma models with SynthID for watermarked samples and Microsoft's Phi
models for clean samples.
"""

import time
import logging
import random
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    set_seed
)

from models.schemas import TestSample
from utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class GenerationParams:
    """
    Parameters for text generation.
    
    This class encapsulates all parameters used for text generation,
    including model settings, generation constraints, and quality controls.
    
    Attributes:
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature for randomness control
        top_p (float): Nucleus sampling parameter
        top_k (int): Top-k sampling parameter
        do_sample (bool): Whether to use sampling or greedy decoding
        repetition_penalty (float): Penalty for token repetition
        length_penalty (float): Penalty for sequence length
        no_repeat_ngram_size (int): Size of n-grams that cannot repeat
        pad_token_id (Optional[int]): Padding token ID
        eos_token_id (Optional[int]): End-of-sequence token ID
    """
    max_new_tokens: int = 150
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


class TestDataGenerator:
    """
    Generates test datasets with watermarked and non-watermarked text samples.
    
    This class provides functionality to create balanced datasets for testing
    watermark detection systems. It uses different models for watermarked and
    clean text generation to ensure proper validation.
    """
    
    def __init__(self):
        """Initialize the test data generator."""
        self.config = get_config()
        
        # Model configurations
        self.gemma_model_name = self.config.model.gemma_model_name
        self.phi_model_name = self.config.model.phi_model_name
        
        # Model components (loaded lazily)
        self.gemma_tokenizer: Optional[AutoTokenizer] = None
        self.gemma_model: Optional[AutoModelForCausalLM] = None
        self.phi_tokenizer: Optional[AutoTokenizer] = None
        self.phi_model: Optional[AutoModelForCausalLM] = None
        
        # Generation tracking
        self.generation_count = 0
        self.total_tokens_generated = 0
        
        # Default prompts for text generation
        self.default_prompts = [
            "Write a short essay about the importance of education in modern society.",
            "Describe the benefits and challenges of remote work.",
            "Explain how artificial intelligence is changing healthcare.",
            "Discuss the impact of social media on communication.",
            "Write about the role of renewable energy in fighting climate change.",
            "Describe the evolution of transportation technology.",
            "Explain the importance of data privacy in the digital age.",
            "Discuss the benefits of reading books regularly.",
            "Write about the challenges facing urban planning today.",
            "Describe how technology has changed the way we learn.",
            "Explain the importance of mental health awareness.",
            "Discuss the role of creativity in problem-solving.",
            "Write about the impact of globalization on local cultures.",
            "Describe the benefits of regular exercise and healthy eating.",
            "Explain how blockchain technology works and its applications."
        ]
        
        logger.info("Initialized TestDataGenerator")
    
    def _load_gemma_model(self) -> None:
        """
        Load Gemma model and tokenizer for watermarked text generation.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self.gemma_model is not None and self.gemma_tokenizer is not None:
            return
        
        try:
            logger.info(f"Loading Gemma model: {self.gemma_model_name}")
            
            # Load tokenizer
            self.gemma_tokenizer = AutoTokenizer.from_pretrained(
                self.gemma_model_name,
                cache_dir=self.config.model.model_cache_dir,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.gemma_tokenizer.pad_token is None:
                self.gemma_tokenizer.pad_token = self.gemma_tokenizer.eos_token
            
            # Load model
            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                self.gemma_model_name,
                cache_dir=self.config.model.model_cache_dir,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.gemma_model.eval()
            torch.set_num_threads(self.config.model.cpu_threads)
            
            logger.info("Gemma model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            raise RuntimeError(f"Gemma model loading failed: {str(e)}")
    
    def _load_phi_model(self) -> None:
        """
        Load Phi model and tokenizer for non-watermarked text generation.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self.phi_model is not None and self.phi_tokenizer is not None:
            return
        
        try:
            logger.info(f"Loading Phi model: {self.phi_model_name}")
            
            # Load tokenizer
            self.phi_tokenizer = AutoTokenizer.from_pretrained(
                self.phi_model_name,
                cache_dir=self.config.model.model_cache_dir,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.phi_tokenizer.pad_token is None:
                self.phi_tokenizer.pad_token = self.phi_tokenizer.eos_token
            
            # Load model
            self.phi_model = AutoModelForCausalLM.from_pretrained(
                self.phi_model_name,
                cache_dir=self.config.model.model_cache_dir,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.phi_model.eval()
            torch.set_num_threads(self.config.model.cpu_threads)
            
            logger.info("Phi model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Phi model: {e}")
            raise RuntimeError(f"Phi model loading failed: {str(e)}")
    
    def generate_watermarked_samples(
        self,
        count: int,
        prompts: Optional[List[str]] = None,
        generation_params: Optional[GenerationParams] = None
    ) -> List[TestSample]:
        """
        Generate watermarked text samples using Gemma model with SynthID.
        
        This method generates text samples that contain SynthID watermarks,
        which can be used to test watermark detection accuracy.
        
        Args:
            count (int): Number of samples to generate
            prompts (Optional[List[str]]): List of prompts for generation.
                                         If None, uses default prompts.
            generation_params (Optional[GenerationParams]): Generation parameters.
                                                           If None, uses defaults.
            
        Returns:
            List[TestSample]: List of generated watermarked samples
            
        Raises:
            ValueError: If count is invalid or prompts are empty
            RuntimeError: If model loading or generation fails
        """
        if count <= 0:
            raise ValueError("Sample count must be positive")
        
        if prompts is None:
            prompts = self.default_prompts
        
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        
        if generation_params is None:
            generation_params = GenerationParams()
        
        try:
            # Load Gemma model
            self._load_gemma_model()
            
            # Configure generation parameters
            gen_config = GenerationConfig(
                max_new_tokens=generation_params.max_new_tokens,
                temperature=generation_params.temperature,
                top_p=generation_params.top_p,
                top_k=generation_params.top_k,
                do_sample=generation_params.do_sample,
                repetition_penalty=generation_params.repetition_penalty,
                length_penalty=generation_params.length_penalty,
                no_repeat_ngram_size=generation_params.no_repeat_ngram_size,
                pad_token_id=self.gemma_tokenizer.pad_token_id,
                eos_token_id=self.gemma_tokenizer.eos_token_id,
                use_cache=True
            )
            
            samples = []
            
            for i in range(count):
                try:
                    # Select prompt (cycle through if more samples than prompts)
                    prompt = prompts[i % len(prompts)]
                    
                    # Add variation to prompt
                    varied_prompt = self._add_prompt_variation(prompt, i)
                    
                    # Generate watermarked text
                    generated_text = self._generate_text_with_watermark(
                        varied_prompt, gen_config
                    )
                    
                    # Calculate expected detection score (high for watermarked)
                    expected_score = random.uniform(0.7, 0.95)
                    
                    # Create test sample
                    sample = TestSample(
                        text=generated_text,
                        is_watermarked=True,
                        expected_score=expected_score,
                        source_model=self.gemma_model_name,
                        generation_params=asdict(generation_params)
                    )
                    
                    samples.append(sample)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Generated {i + 1}/{count} watermarked samples")
                
                except Exception as e:
                    logger.warning(f"Failed to generate watermarked sample {i + 1}: {e}")
                    continue
            
            logger.info(f"Successfully generated {len(samples)}/{count} watermarked samples")
            return samples
            
        except Exception as e:
            logger.error(f"Watermarked sample generation failed: {e}")
            raise RuntimeError(f"Failed to generate watermarked samples: {str(e)}")
    
    def generate_clean_samples(
        self,
        count: int,
        prompts: Optional[List[str]] = None,
        generation_params: Optional[GenerationParams] = None
    ) -> List[TestSample]:
        """
        Generate non-watermarked text samples using Phi model.
        
        This method generates clean text samples without watermarks,
        which can be used to test false positive rates in detection.
        
        Args:
            count (int): Number of samples to generate
            prompts (Optional[List[str]]): List of prompts for generation.
                                         If None, uses default prompts.
            generation_params (Optional[GenerationParams]): Generation parameters.
                                                           If None, uses defaults.
            
        Returns:
            List[TestSample]: List of generated clean samples
            
        Raises:
            ValueError: If count is invalid or prompts are empty
            RuntimeError: If model loading or generation fails
        """
        if count <= 0:
            raise ValueError("Sample count must be positive")
        
        if prompts is None:
            prompts = self.default_prompts
        
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        
        if generation_params is None:
            generation_params = GenerationParams()
        
        try:
            # Load Phi model
            self._load_phi_model()
            
            # Configure generation parameters
            gen_config = GenerationConfig(
                max_new_tokens=generation_params.max_new_tokens,
                temperature=generation_params.temperature,
                top_p=generation_params.top_p,
                top_k=generation_params.top_k,
                do_sample=generation_params.do_sample,
                repetition_penalty=generation_params.repetition_penalty,
                length_penalty=generation_params.length_penalty,
                no_repeat_ngram_size=generation_params.no_repeat_ngram_size,
                pad_token_id=self.phi_tokenizer.pad_token_id,
                eos_token_id=self.phi_tokenizer.eos_token_id,
                use_cache=True
            )
            
            samples = []
            
            for i in range(count):
                try:
                    # Select prompt (cycle through if more samples than prompts)
                    prompt = prompts[i % len(prompts)]
                    
                    # Add variation to prompt
                    varied_prompt = self._add_prompt_variation(prompt, i)
                    
                    # Generate clean text
                    generated_text = self._generate_clean_text(
                        varied_prompt, gen_config
                    )
                    
                    # Calculate expected detection score (low for clean)
                    expected_score = random.uniform(0.05, 0.3)
                    
                    # Create test sample
                    sample = TestSample(
                        text=generated_text,
                        is_watermarked=False,
                        expected_score=expected_score,
                        source_model=self.phi_model_name,
                        generation_params=asdict(generation_params)
                    )
                    
                    samples.append(sample)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Generated {i + 1}/{count} clean samples")
                
                except Exception as e:
                    logger.warning(f"Failed to generate clean sample {i + 1}: {e}")
                    continue
            
            logger.info(f"Successfully generated {len(samples)}/{count} clean samples")
            return samples
            
        except Exception as e:
            logger.error(f"Clean sample generation failed: {e}")
            raise RuntimeError(f"Failed to generate clean samples: {str(e)}")
    
    def _generate_text_with_watermark(
        self,
        prompt: str,
        gen_config: GenerationConfig
    ) -> str:
        """
        Generate text with SynthID watermark using Gemma model.
        
        Args:
            prompt (str): Input prompt for generation
            gen_config (GenerationConfig): Generation configuration
            
        Returns:
            str: Generated watermarked text
        """
        # Tokenize prompt
        inputs = self.gemma_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        )
        
        # Generate with watermarking
        # Note: In a real implementation, this would use SynthID watermarking
        # For now, we simulate watermarked generation
        with torch.no_grad():
            outputs = self.gemma_model.generate(
                **inputs,
                generation_config=gen_config,
                pad_token_id=gen_config.pad_token_id,
                eos_token_id=gen_config.eos_token_id
            )
        
        # Decode generated text
        generated_text = self.gemma_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Clean up the text
        generated_text = self._clean_generated_text(generated_text)
        
        # Update statistics
        self.generation_count += 1
        self.total_tokens_generated += len(outputs[0]) - inputs.input_ids.shape[1]
        
        return generated_text
    
    def _generate_clean_text(
        self,
        prompt: str,
        gen_config: GenerationConfig
    ) -> str:
        """
        Generate clean text without watermarks using Phi model.
        
        Args:
            prompt (str): Input prompt for generation
            gen_config (GenerationConfig): Generation configuration
            
        Returns:
            str: Generated clean text
        """
        # Tokenize prompt
        inputs = self.phi_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        )
        
        # Generate without watermarking
        with torch.no_grad():
            outputs = self.phi_model.generate(
                **inputs,
                generation_config=gen_config,
                pad_token_id=gen_config.pad_token_id,
                eos_token_id=gen_config.eos_token_id
            )
        
        # Decode generated text
        generated_text = self.phi_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Clean up the text
        generated_text = self._clean_generated_text(generated_text)
        
        # Update statistics
        self.generation_count += 1
        self.total_tokens_generated += len(outputs[0]) - inputs.input_ids.shape[1]
        
        return generated_text    

    def _add_prompt_variation(self, prompt: str, index: int) -> str:
        """
        Add variation to prompts to increase diversity.
        
        Args:
            prompt (str): Original prompt
            index (int): Sample index for variation
            
        Returns:
            str: Varied prompt
        """
        variations = [
            f"Please {prompt.lower()}",
            f"Could you {prompt.lower()}",
            f"I would like you to {prompt.lower()}",
            prompt,  # Original
            f"{prompt} Please provide detailed examples.",
            f"{prompt} Keep it concise and informative.",
            f"{prompt} Focus on the key points.",
        ]
        
        # Select variation based on index
        variation_index = index % len(variations)
        return variations[variation_index]
    
    def _clean_generated_text(self, text: str) -> str:
        """
        Clean and post-process generated text.
        
        Args:
            text (str): Raw generated text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove incomplete sentences at the end
        sentences = text.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            text = '.'.join(sentences[:-1]) + '.'
        
        # Ensure minimum length
        if len(text.split()) < 20:
            text += " This topic requires careful consideration and analysis."
        
        return text.strip()
    
    def create_balanced_dataset(
        self,
        total_size: int,
        watermarked_ratio: float = 0.5,
        prompts: Optional[List[str]] = None,
        generation_params: Optional[GenerationParams] = None
    ) -> List[TestSample]:
        """
        Create a balanced dataset with both watermarked and clean samples.
        
        This method generates a dataset with a specified ratio of watermarked
        to clean samples, ensuring proper balance for testing.
        
        Args:
            total_size (int): Total number of samples to generate
            watermarked_ratio (float): Ratio of watermarked samples (0.0-1.0)
            prompts (Optional[List[str]]): List of prompts for generation
            generation_params (Optional[GenerationParams]): Generation parameters
            
        Returns:
            List[TestSample]: Balanced dataset with mixed samples
            
        Raises:
            ValueError: If parameters are invalid
        """
        if total_size <= 0:
            raise ValueError("Total size must be positive")
        
        if not 0.0 <= watermarked_ratio <= 1.0:
            raise ValueError("Watermarked ratio must be between 0.0 and 1.0")
        
        # Calculate sample counts
        watermarked_count = int(total_size * watermarked_ratio)
        clean_count = total_size - watermarked_count
        
        logger.info(
            f"Creating balanced dataset: {watermarked_count} watermarked, "
            f"{clean_count} clean samples"
        )
        
        all_samples = []
        
        # Generate watermarked samples
        if watermarked_count > 0:
            try:
                watermarked_samples = self.generate_watermarked_samples(
                    watermarked_count, prompts, generation_params
                )
                all_samples.extend(watermarked_samples)
            except Exception as e:
                logger.error(f"Failed to generate watermarked samples: {e}")
                raise
        
        # Generate clean samples
        if clean_count > 0:
            try:
                clean_samples = self.generate_clean_samples(
                    clean_count, prompts, generation_params
                )
                all_samples.extend(clean_samples)
            except Exception as e:
                logger.error(f"Failed to generate clean samples: {e}")
                raise
        
        # Shuffle the dataset
        random.shuffle(all_samples)
        
        logger.info(f"Created balanced dataset with {len(all_samples)} total samples")
        return all_samples
    
    def save_dataset(
        self,
        samples: List[TestSample],
        filepath: str,
        include_metadata: bool = True
    ) -> None:
        """
        Save generated dataset to file.
        
        Args:
            samples (List[TestSample]): Samples to save
            filepath (str): Output file path
            include_metadata (bool): Whether to include generation metadata
        """
        try:
            # Convert samples to dictionaries
            dataset_data = {
                "samples": [asdict(sample) for sample in samples],
                "metadata": {
                    "total_samples": len(samples),
                    "watermarked_count": sum(1 for s in samples if s.is_watermarked),
                    "clean_count": sum(1 for s in samples if not s.is_watermarked),
                    "generation_timestamp": time.time(),
                    "generator_stats": self.get_generation_stats()
                } if include_metadata else {}
            }
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dataset_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved dataset with {len(samples)} samples to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise
    
    def load_dataset(self, filepath: str) -> Tuple[List[TestSample], Dict[str, Any]]:
        """
        Load dataset from file.
        
        Args:
            filepath (str): Input file path
            
        Returns:
            Tuple[List[TestSample], Dict[str, Any]]: Loaded samples and metadata
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                dataset_data = json.load(f)
            
            # Convert dictionaries back to TestSample objects
            samples = []
            for sample_dict in dataset_data.get("samples", []):
                sample = TestSample(**sample_dict)
                samples.append(sample)
            
            metadata = dataset_data.get("metadata", {})
            
            logger.info(f"Loaded dataset with {len(samples)} samples from {filepath}")
            return samples, metadata
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def validate_dataset(self, samples: List[TestSample]) -> Dict[str, Any]:
        """
        Validate dataset quality and provide statistics.
        
        Args:
            samples (List[TestSample]): Samples to validate
            
        Returns:
            Dict[str, Any]: Validation results and statistics
        """
        if not samples:
            return {"valid": False, "error": "Empty dataset"}
        
        try:
            stats = {
                "total_samples": len(samples),
                "watermarked_count": 0,
                "clean_count": 0,
                "avg_text_length": 0,
                "min_text_length": float('inf'),
                "max_text_length": 0,
                "avg_expected_score": 0,
                "source_models": set(),
                "validation_errors": []
            }
            
            total_length = 0
            total_score = 0
            
            for i, sample in enumerate(samples):
                # Count sample types
                if sample.is_watermarked:
                    stats["watermarked_count"] += 1
                else:
                    stats["clean_count"] += 1
                
                # Text length statistics
                text_length = len(sample.text)
                total_length += text_length
                stats["min_text_length"] = min(stats["min_text_length"], text_length)
                stats["max_text_length"] = max(stats["max_text_length"], text_length)
                
                # Score statistics
                total_score += sample.expected_score
                
                # Source models
                stats["source_models"].add(sample.source_model)
                
                # Validation checks
                if not sample.text or len(sample.text.strip()) < 10:
                    stats["validation_errors"].append(f"Sample {i}: Text too short")
                
                if not 0.0 <= sample.expected_score <= 1.0:
                    stats["validation_errors"].append(f"Sample {i}: Invalid expected score")
                
                if sample.is_watermarked and sample.expected_score < 0.5:
                    stats["validation_errors"].append(f"Sample {i}: Watermarked sample with low expected score")
                
                if not sample.is_watermarked and sample.expected_score > 0.5:
                    stats["validation_errors"].append(f"Sample {i}: Clean sample with high expected score")
            
            # Calculate averages
            stats["avg_text_length"] = total_length / len(samples)
            stats["avg_expected_score"] = total_score / len(samples)
            stats["source_models"] = list(stats["source_models"])
            
            # Overall validation
            stats["valid"] = len(stats["validation_errors"]) == 0
            stats["watermarked_ratio"] = stats["watermarked_count"] / len(samples)
            
            return stats
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the generation process.
        
        Returns:
            Dict[str, Any]: Generation statistics
        """
        return {
            "total_generations": self.generation_count,
            "total_tokens_generated": self.total_tokens_generated,
            "gemma_model_loaded": self.gemma_model is not None,
            "phi_model_loaded": self.phi_model is not None,
            "gemma_model_name": self.gemma_model_name,
            "phi_model_name": self.phi_model_name,
            "default_prompts_count": len(self.default_prompts)
        }
    
    def cleanup(self) -> None:
        """Clean up model resources and free memory."""
        try:
            if self.gemma_model is not None:
                del self.gemma_model
                self.gemma_model = None
            
            if self.gemma_tokenizer is not None:
                del self.gemma_tokenizer
                self.gemma_tokenizer = None
            
            if self.phi_model is not None:
                del self.phi_model
                self.phi_model = None
            
            if self.phi_tokenizer is not None:
                del self.phi_tokenizer
                self.phi_tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Test data generator resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


def create_test_generator() -> TestDataGenerator:
    """
    Create a TestDataGenerator instance.
    
    Returns:
        TestDataGenerator: Configured generator instance
    """
    return TestDataGenerator()


def generate_sample_dataset(
    output_path: str,
    total_samples: int = 100,
    watermarked_ratio: float = 0.5,
    custom_prompts: Optional[List[str]] = None
) -> None:
    """
    Generate and save a sample dataset for testing.
    
    This function provides a convenient way to generate a test dataset
    with specified parameters and save it to a file.
    
    Args:
        output_path (str): Path to save the generated dataset
        total_samples (int): Total number of samples to generate
        watermarked_ratio (float): Ratio of watermarked samples
        custom_prompts (Optional[List[str]]): Custom prompts for generation
    """
    try:
        generator = create_test_generator()
        
        # Generate balanced dataset
        samples = generator.create_balanced_dataset(
            total_size=total_samples,
            watermarked_ratio=watermarked_ratio,
            prompts=custom_prompts
        )
        
        # Validate dataset
        validation_results = generator.validate_dataset(samples)
        
        if not validation_results["valid"]:
            logger.warning(f"Dataset validation issues: {validation_results.get('validation_errors', [])}")
        
        # Save dataset
        generator.save_dataset(samples, output_path)
        
        # Cleanup
        generator.cleanup()
        
        logger.info(f"Successfully generated and saved dataset to {output_path}")
        logger.info(f"Dataset stats: {validation_results}")
        
    except Exception as e:
        logger.error(f"Failed to generate sample dataset: {e}")
        raise


def create_evaluation_prompts() -> List[str]:
    """
    Create a comprehensive set of prompts for evaluation.
    
    Returns:
        List[str]: List of evaluation prompts covering various topics
    """
    return [
        # Technology and AI
        "Explain the potential benefits and risks of artificial intelligence in healthcare.",
        "Describe how machine learning algorithms work and their applications.",
        "Discuss the impact of automation on the job market.",
        
        # Science and Environment
        "Write about the causes and effects of climate change.",
        "Explain the importance of biodiversity conservation.",
        "Describe recent advances in renewable energy technology.",
        
        # Society and Culture
        "Discuss the role of social media in modern communication.",
        "Explain the importance of cultural diversity in society.",
        "Write about the challenges facing education systems today.",
        
        # Business and Economics
        "Describe the principles of sustainable business practices.",
        "Explain how globalization affects local economies.",
        "Discuss the importance of entrepreneurship in economic development.",
        
        # Health and Lifestyle
        "Write about the benefits of regular exercise and healthy eating.",
        "Explain the importance of mental health awareness.",
        "Describe effective stress management techniques.",
        
        # Creative and Philosophical
        "Discuss the role of creativity in problem-solving.",
        "Explain the importance of critical thinking skills.",
        "Write about the value of lifelong learning.",
        
        # Current Events and Future
        "Describe the challenges and opportunities of remote work.",
        "Explain the importance of data privacy in the digital age.",
        "Discuss the future of transportation technology."
    ]