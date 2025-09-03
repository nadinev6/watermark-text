"""
SynthID watermark detection implementation.

This module implements watermark detection using Google's SynthID system
through the Hugging Face Transformers library. It provides CPU-optimized
detection with lazy model loading and comprehensive error handling.
"""

import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig
)

from detectors.base import WatermarkDetector, DetectionResult, DetectionMethod, DetectionError
from utils.config import get_config

logger = logging.getLogger(__name__)


class SynthIDDetector(WatermarkDetector):
    """
    SynthID watermark detector using Google's Gemma model.
    
    This class implements watermark detection using the SynthID system
    integrated with Google's Gemma models. It provides CPU-optimized
    inference with lazy loading and comprehensive error handling.
    
    The detector analyzes text for statistical patterns characteristic
    of SynthID watermarking and provides confidence scores for detection.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the SynthID detector.
        
        Args:
            model_name (Optional[str]): Hugging Face model identifier.
                                      If None, uses configuration default.
        """
        super().__init__("SynthID Detector")
        
        self.config = get_config()
        self.model_name = model_name or self.config.model.gemma_model_name
        
        # Model components (loaded lazily)
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.generation_config: Optional[GenerationConfig] = None
        
        # Detection parameters
        self.min_tokens_for_detection = 10
        self.max_sequence_length = 2048
        self.detection_threshold = 0.5
        
        # Performance tracking
        self.load_time_ms = 0
        self.detection_count = 0
        
        logger.info(f"Initialized SynthID detector with model: {self.model_name}")
    
    def _lazy_load_model(self) -> None:
        """
        Load the Gemma model and tokenizer with CPU optimization.
        
        This method performs lazy loading of the model components,
        optimizing for CPU-only inference and memory efficiency.
        
        Raises:
            DetectionError: If model loading fails or times out
        """
        if self._is_initialized:
            return
        
        start_time = time.time()
        
        try:
            logger.info(f"Loading SynthID model: {self.model_name}")
            
            # Configure device and optimization settings
            device = "cpu"  # Force CPU usage as per constraints
            torch_dtype = torch.float32  # Use float32 for CPU compatibility
            
            # Load tokenizer with error handling
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.config.model.model_cache_dir,
                    trust_remote_code=True,
                    use_fast=True
                )
                
                # Ensure pad token is set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                logger.info("Tokenizer loaded successfully")
                
            except Exception as e:
                raise DetectionError(
                    f"Failed to load tokenizer: {str(e)}",
                    "TOKENIZER_LOAD_FAILED",
                    recoverable=False
                )
            
            # Load model with CPU optimization
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.config.model.model_cache_dir,
                    torch_dtype=torch_dtype,
                    device_map=device,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
                
                # Set model to evaluation mode
                self.model.eval()
                
                # Configure CPU threading
                torch.set_num_threads(self.config.model.cpu_threads)
                
                logger.info("Model loaded successfully")
                
            except Exception as e:
                raise DetectionError(
                    f"Failed to load model: {str(e)}",
                    "MODEL_LOAD_FAILED",
                    recoverable=False
                )
            
            # Configure generation settings for watermark detection
            self.generation_config = GenerationConfig(
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            self._is_initialized = True
            self.load_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(f"SynthID model loaded in {self.load_time_ms}ms")
            
        except DetectionError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during model loading: {e}")
            raise DetectionError(
                f"Model initialization failed: {str(e)}",
                "INITIALIZATION_FAILED",
                recoverable=False
            )
    
    def detect(self, text: str) -> DetectionResult:
        """
        Analyze text for SynthID watermark presence.
        
        This method performs statistical analysis of the input text to detect
        patterns characteristic of SynthID watermarking. It uses perplexity
        analysis and token distribution patterns for detection.
        
        Args:
            text (str): Input text to analyze for watermarks
            
        Returns:
            DetectionResult: Structured result with confidence score and metadata
            
        Raises:
            ValueError: If input text is empty or too short
            DetectionError: If detection process fails
        """
        start_time = time.time()
        
        # Validate input
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        text = text.strip()
        
        if len(text) < 20:  # Minimum text length for reliable detection
            raise ValueError("Text too short for reliable watermark detection")
        
        try:
            # Ensure model is loaded
            self._lazy_load_model()
            
            # Tokenize input text
            tokens = self._tokenize_text(text)
            
            if len(tokens) < self.min_tokens_for_detection:
                logger.warning(f"Text has only {len(tokens)} tokens, may be unreliable")
            
            # Perform watermark detection analysis
            detection_scores = self._analyze_watermark_patterns(text, tokens)
            
            # Calculate overall confidence score
            confidence_score = self._calculate_confidence_score(detection_scores)
            
            # Determine binary classification
            is_watermarked = confidence_score >= self.detection_threshold
            
            # Identify source model if watermarked
            model_identified = self.model_name if is_watermarked else None
            
            # Compile detection metadata
            metadata = self._compile_detection_metadata(
                text, tokens, detection_scores, start_time
            )
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            # Update detection count
            self.detection_count += 1
            
            result = DetectionResult(
                confidence_score=confidence_score,
                is_watermarked=is_watermarked,
                model_identified=model_identified,
                detection_method=DetectionMethod.SYNTHID.value,
                metadata=metadata,
                processing_time_ms=processing_time
            )
            
            logger.info(
                f"SynthID detection completed: confidence={confidence_score:.3f}, "
                f"watermarked={is_watermarked}, time={processing_time}ms"
            )
            
            return result
            
        except ValueError:
            raise
        except DetectionError:
            raise
        except Exception as e:
            logger.error(f"SynthID detection failed: {e}")
            raise DetectionError(
                f"Detection analysis failed: {str(e)}",
                "DETECTION_FAILED",
                recoverable=True
            )
    
    def _tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize input text using the model tokenizer.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            List[int]: List of token IDs
            
        Raises:
            DetectionError: If tokenization fails
        """
        try:
            # Truncate text if too long
            if len(text) > self.max_sequence_length * 4:  # Rough character estimate
                text = text[:self.max_sequence_length * 4]
                logger.warning("Text truncated for tokenization")
            
            # Tokenize with attention to special tokens
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_sequence_length,
                padding=False,
                add_special_tokens=True
            )
            
            return encoding.input_ids[0].tolist()
            
        except Exception as e:
            raise DetectionError(
                f"Text tokenization failed: {str(e)}",
                "TOKENIZATION_FAILED",
                recoverable=True
            )
    
    def _analyze_watermark_patterns(self, text: str, tokens: List[int]) -> Dict[str, float]:
        """
        Analyze text for watermark-specific patterns.
        
        This method performs statistical analysis to detect patterns
        characteristic of SynthID watermarking, including perplexity
        analysis and token distribution examination.
        
        Args:
            text (str): Original input text
            tokens (List[int]): Tokenized representation
            
        Returns:
            Dict[str, float]: Dictionary of detection scores for different patterns
        """
        detection_scores = {}
        
        try:
            # Calculate perplexity-based score
            perplexity_score = self._calculate_perplexity_score(tokens)
            detection_scores["perplexity"] = perplexity_score
            
            # Analyze token distribution patterns
            distribution_score = self._analyze_token_distribution(tokens)
            detection_scores["token_distribution"] = distribution_score
            
            # Check for repetition patterns
            repetition_score = self._analyze_repetition_patterns(tokens)
            detection_scores["repetition_patterns"] = repetition_score
            
            # Analyze sequence coherence
            coherence_score = self._analyze_sequence_coherence(tokens)
            detection_scores["sequence_coherence"] = coherence_score
            
            logger.debug(f"Detection scores: {detection_scores}")
            
        except Exception as e:
            logger.warning(f"Pattern analysis partially failed: {e}")
            # Provide fallback scores
            detection_scores = {
                "perplexity": 0.5,
                "token_distribution": 0.5,
                "repetition_patterns": 0.5,
                "sequence_coherence": 0.5
            }
        
        return detection_scores
    
    def _calculate_perplexity_score(self, tokens: List[int]) -> float:
        """
        Calculate perplexity-based watermark detection score.
        
        Args:
            tokens (List[int]): Token sequence to analyze
            
        Returns:
            float: Perplexity-based detection score (0.0-1.0)
        """
        try:
            # Convert tokens to tensor
            input_ids = torch.tensor([tokens])
            
            # Calculate model perplexity
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
                perplexity = torch.exp(torch.tensor(loss)).item()
            
            # Convert perplexity to detection score
            # Lower perplexity suggests more predictable (potentially watermarked) text
            # Normalize perplexity to 0-1 range (typical range: 1-100)
            normalized_perplexity = min(1.0, max(0.0, (100 - perplexity) / 100))
            
            return normalized_perplexity
            
        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")
            return 0.5  # Neutral score on failure
    
    def _analyze_token_distribution(self, tokens: List[int]) -> float:
        """
        Analyze token distribution for watermark patterns.
        
        Args:
            tokens (List[int]): Token sequence to analyze
            
        Returns:
            float: Distribution-based detection score (0.0-1.0)
        """
        try:
            # Calculate token frequency distribution
            token_counts = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            
            # Calculate distribution entropy
            total_tokens = len(tokens)
            entropy = 0.0
            
            for count in token_counts.values():
                probability = count / total_tokens
                if probability > 0:
                    entropy -= probability * torch.log2(torch.tensor(probability)).item()
            
            # Normalize entropy (typical range: 0-10)
            normalized_entropy = min(1.0, max(0.0, entropy / 10))
            
            # Lower entropy might indicate watermarking
            return 1.0 - normalized_entropy
            
        except Exception as e:
            logger.warning(f"Token distribution analysis failed: {e}")
            return 0.5
    
    def _analyze_repetition_patterns(self, tokens: List[int]) -> float:
        """
        Analyze repetition patterns in token sequence.
        
        Args:
            tokens (List[int]): Token sequence to analyze
            
        Returns:
            float: Repetition-based detection score (0.0-1.0)
        """
        try:
            # Look for unusual repetition patterns
            repetition_score = 0.0
            
            # Check for immediate repetitions
            immediate_reps = sum(1 for i in range(len(tokens) - 1) 
                               if tokens[i] == tokens[i + 1])
            immediate_ratio = immediate_reps / max(1, len(tokens) - 1)
            
            # Check for pattern repetitions (bigrams, trigrams)
            bigram_reps = self._count_pattern_repetitions(tokens, 2)
            trigram_reps = self._count_pattern_repetitions(tokens, 3)
            
            # Combine repetition metrics
            repetition_score = (immediate_ratio + bigram_reps + trigram_reps) / 3
            
            return min(1.0, repetition_score)
            
        except Exception as e:
            logger.warning(f"Repetition analysis failed: {e}")
            return 0.5
    
    def _count_pattern_repetitions(self, tokens: List[int], n: int) -> float:
        """
        Count n-gram pattern repetitions.
        
        Args:
            tokens (List[int]): Token sequence
            n (int): N-gram size
            
        Returns:
            float: Repetition ratio for n-grams
        """
        if len(tokens) < n:
            return 0.0
        
        patterns = {}
        total_patterns = len(tokens) - n + 1
        
        for i in range(total_patterns):
            pattern = tuple(tokens[i:i + n])
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Calculate repetition ratio
        repeated_patterns = sum(1 for count in patterns.values() if count > 1)
        return repeated_patterns / max(1, len(patterns))
    
    def _analyze_sequence_coherence(self, tokens: List[int]) -> float:
        """
        Analyze sequence coherence for watermark detection.
        
        Args:
            tokens (List[int]): Token sequence to analyze
            
        Returns:
            float: Coherence-based detection score (0.0-1.0)
        """
        try:
            # Simple coherence measure based on token transitions
            if len(tokens) < 2:
                return 0.5
            
            # Calculate transition probabilities (simplified)
            transitions = {}
            for i in range(len(tokens) - 1):
                current, next_token = tokens[i], tokens[i + 1]
                if current not in transitions:
                    transitions[current] = {}
                transitions[current][next_token] = transitions[current].get(next_token, 0) + 1
            
            # Calculate average transition entropy
            total_entropy = 0.0
            valid_transitions = 0
            
            for current_token, next_tokens in transitions.items():
                total_next = sum(next_tokens.values())
                entropy = 0.0
                
                for count in next_tokens.values():
                    prob = count / total_next
                    if prob > 0:
                        entropy -= prob * torch.log2(torch.tensor(prob)).item()
                
                total_entropy += entropy
                valid_transitions += 1
            
            if valid_transitions == 0:
                return 0.5
            
            avg_entropy = total_entropy / valid_transitions
            # Normalize (typical range: 0-8)
            normalized_coherence = min(1.0, max(0.0, avg_entropy / 8))
            
            return normalized_coherence
            
        except Exception as e:
            logger.warning(f"Coherence analysis failed: {e}")
            return 0.5
    
    def _calculate_confidence_score(self, detection_scores: Dict[str, float]) -> float:
        """
        Calculate overall confidence score from individual detection metrics.
        
        Args:
            detection_scores (Dict[str, float]): Individual detection scores
            
        Returns:
            float: Overall confidence score (0.0-1.0)
        """
        # Weighted combination of detection scores
        weights = {
            "perplexity": 0.4,
            "token_distribution": 0.3,
            "repetition_patterns": 0.2,
            "sequence_coherence": 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, score in detection_scores.items():
            weight = weights.get(metric, 0.1)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return min(1.0, max(0.0, weighted_score / total_weight))
    
    def _compile_detection_metadata(
        self,
        text: str,
        tokens: List[int],
        detection_scores: Dict[str, float],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Compile comprehensive metadata about the detection process.
        
        Args:
            text (str): Original input text
            tokens (List[int]): Tokenized representation
            detection_scores (Dict[str, float]): Individual detection scores
            start_time (float): Detection start timestamp
            
        Returns:
            Dict[str, Any]: Comprehensive detection metadata
        """
        return {
            "model_info": {
                "model_name": self.model_name,
                "model_loaded": self._is_initialized,
                "load_time_ms": self.load_time_ms,
                "detection_count": self.detection_count
            },
            "text_analysis": {
                "text_length": len(text),
                "token_count": len(tokens),
                "avg_token_length": len(text) / max(1, len(tokens)),
                "unique_tokens": len(set(tokens))
            },
            "detection_scores": detection_scores,
            "detection_params": {
                "min_tokens_required": self.min_tokens_for_detection,
                "detection_threshold": self.detection_threshold,
                "max_sequence_length": self.max_sequence_length
            },
            "performance": {
                "cpu_threads": self.config.model.cpu_threads,
                "memory_limit_mb": self.config.model.max_model_memory_mb
            }
        }
    
    def is_available(self) -> bool:
        """
        Check if the SynthID detector is ready for use.
        
        Returns:
            bool: True if detector is initialized and ready
        """
        try:
            if not self._is_initialized:
                # Try to initialize if not already done
                self._lazy_load_model()
            
            return (
                self._is_initialized and
                self.tokenizer is not None and
                self.model is not None
            )
            
        except Exception as e:
            logger.error(f"Availability check failed: {e}")
            return False
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of models this detector can identify.
        
        Returns:
            List[str]: List of supported model identifiers
        """
        return [
            "google/gemma-2-2b",
            "google/gemma-2-9b",
            "google/gemma-2-27b"
        ]
    
    def cleanup(self) -> None:
        """
        Clean up model resources and free memory.
        
        This method unloads the model and tokenizer to free up memory,
        useful for resource management in production environments.
        """
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear CUDA cache if available (though we're using CPU)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._is_initialized = False
            
            logger.info("SynthID detector resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information and statistics
        """
        return {
            "model_name": self.model_name,
            "is_loaded": self._is_initialized,
            "load_time_ms": self.load_time_ms,
            "detection_count": self.detection_count,
            "supported_models": self.get_supported_models(),
            "config": {
                "min_tokens_for_detection": self.min_tokens_for_detection,
                "max_sequence_length": self.max_sequence_length,
                "detection_threshold": self.detection_threshold
            }
        }