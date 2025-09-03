"""
Custom fallback watermark detection implementation.

This module implements a custom watermark detection system that serves as a
fallback when SynthID detection is unavailable or inconclusive. It uses
statistical analysis, perplexity measurements, and linguistic features to
identify AI-generated text patterns.
"""

import time
import logging
import math
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter, defaultdict
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from detectors.base import WatermarkDetector, DetectionResult, DetectionMethod, DetectionError
from utils.config import get_config

logger = logging.getLogger(__name__)


class CustomDetector(WatermarkDetector):
    """
    Custom fallback watermark detector using statistical analysis.
    
    This class implements a fallback detection system that analyzes text
    for patterns characteristic of AI generation when SynthID detection
    is unavailable. It uses multiple statistical measures and linguistic
    features to identify AI-generated content.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the custom detector.
        
        Args:
            model_name (Optional[str]): Hugging Face model identifier for reference.
                                      If None, uses Phi model from configuration.
        """
        super().__init__("Custom Fallback Detector")
        
        self.config = get_config()
        self.model_name = model_name or self.config.model.phi_model_name
        
        # Model components (loaded lazily for perplexity analysis)
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        
        # Detection parameters
        self.min_words_for_detection = 20
        self.max_sequence_length = 1024
        
        # Statistical thresholds (tuned for AI detection)
        self.perplexity_threshold = 15.0  # Lower perplexity suggests AI generation
        self.repetition_threshold = 0.15  # Higher repetition suggests AI
        self.burstiness_threshold = 0.3   # Lower burstiness suggests AI
        self.entropy_threshold = 4.5      # Different entropy patterns in AI text
        
        # Performance tracking
        self.detection_count = 0
        self.load_time_ms = 0
        
        logger.info(f"Initialized Custom detector with reference model: {self.model_name}")
    
    def _lazy_load_model(self) -> None:
        """
        Load the reference model for perplexity analysis (optional).
        
        This method loads a reference model that can be used for perplexity
        calculations. If loading fails, the detector will still work using
        other statistical measures.
        
        Raises:
            DetectionError: If model loading fails critically
        """
        if self._is_initialized:
            return
        
        start_time = time.time()
        
        try:
            logger.info(f"Loading reference model for perplexity analysis: {self.model_name}")
            
            # Configure for CPU usage
            device = "cpu"
            torch_dtype = torch.float32
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.config.model.model_cache_dir,
                    trust_remote_code=True,
                    use_fast=True
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                logger.info("Reference tokenizer loaded successfully")
                
            except Exception as e:
                logger.warning(f"Failed to load reference tokenizer: {e}")
                # Continue without tokenizer - other methods will still work
                self.tokenizer = None
            
            # Load model (optional for perplexity analysis)
            try:
                if self.tokenizer is not None:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.config.model.model_cache_dir,
                        torch_dtype=torch_dtype,
                        device_map=device,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    
                    self.model.eval()
                    torch.set_num_threads(self.config.model.cpu_threads)
                    
                    logger.info("Reference model loaded successfully")
                
            except Exception as e:
                logger.warning(f"Failed to load reference model: {e}")
                # Continue without model - other detection methods will work
                self.model = None
            
            self._is_initialized = True
            self.load_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(f"Custom detector initialized in {self.load_time_ms}ms")
            
        except Exception as e:
            logger.error(f"Critical error during custom detector initialization: {e}")
            # Set as initialized anyway - statistical methods don't require models
            self._is_initialized = True
            self.load_time_ms = int((time.time() - start_time) * 1000)
    
    def detect(self, text: str) -> DetectionResult:
        """
        Analyze text for AI-generated patterns using statistical methods.
        
        This method performs comprehensive statistical analysis to detect
        patterns characteristic of AI-generated text, including perplexity,
        repetition patterns, burstiness, and linguistic features.
        
        Args:
            text (str): Input text to analyze for AI generation patterns
            
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
        words = text.split()
        
        if len(words) < self.min_words_for_detection:
            raise ValueError(f"Text too short for reliable detection: need at least {self.min_words_for_detection} words")
        
        try:
            # Ensure detector is initialized
            self._lazy_load_model()
            
            # Perform statistical analysis
            analysis_results = self._perform_statistical_analysis(text, words)
            
            # Calculate overall confidence score
            confidence_score = self._calculate_confidence_score(analysis_results)
            
            # Determine binary classification
            is_ai_generated = confidence_score >= 0.5
            
            # Model identification (generic for custom detector)
            model_identified = "ai-generated" if is_ai_generated else None
            
            # Compile detection metadata
            metadata = self._compile_detection_metadata(
                text, words, analysis_results, start_time
            )
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            # Update detection count
            self.detection_count += 1
            
            result = DetectionResult(
                confidence_score=confidence_score,
                is_watermarked=is_ai_generated,  # For custom detector, this means AI-generated
                model_identified=model_identified,
                detection_method=DetectionMethod.CUSTOM.value,
                metadata=metadata,
                processing_time_ms=processing_time
            )
            
            logger.info(
                f"Custom detection completed: confidence={confidence_score:.3f}, "
                f"ai_generated={is_ai_generated}, time={processing_time}ms"
            )
            
            return result
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Custom detection failed: {e}")
            raise DetectionError(
                f"Statistical analysis failed: {str(e)}",
                "CUSTOM_DETECTION_FAILED",
                recoverable=True
            )
    
    def _perform_statistical_analysis(self, text: str, words: List[str]) -> Dict[str, float]:
        """
        Perform comprehensive statistical analysis of the text.
        
        Args:
            text (str): Original input text
            words (List[str]): List of words in the text
            
        Returns:
            Dict[str, float]: Dictionary of analysis results and scores
        """
        analysis_results = {}
        
        try:
            # Perplexity analysis (if model is available)
            if self.model is not None and self.tokenizer is not None:
                perplexity_score = self._calculate_perplexity_score(text)
                analysis_results["perplexity"] = perplexity_score
            else:
                analysis_results["perplexity"] = 0.5  # Neutral score if no model
            
            # Repetition pattern analysis
            repetition_score = self._analyze_repetition_patterns(words)
            analysis_results["repetition"] = repetition_score
            
            # Burstiness analysis
            burstiness_score = self._analyze_burstiness(words)
            analysis_results["burstiness"] = burstiness_score
            
            # Entropy analysis
            entropy_score = self._analyze_entropy(words)
            analysis_results["entropy"] = entropy_score
            
            # Linguistic feature analysis
            linguistic_score = self._analyze_linguistic_features(text, words)
            analysis_results["linguistic"] = linguistic_score
            
            # Sentence structure analysis
            structure_score = self._analyze_sentence_structure(text)
            analysis_results["structure"] = structure_score
            
            # Vocabulary diversity analysis
            diversity_score = self._analyze_vocabulary_diversity(words)
            analysis_results["diversity"] = diversity_score
            
            logger.debug(f"Statistical analysis results: {analysis_results}")
            
        except Exception as e:
            logger.warning(f"Statistical analysis partially failed: {e}")
            # Provide fallback neutral scores
            analysis_results = {
                "perplexity": 0.5,
                "repetition": 0.5,
                "burstiness": 0.5,
                "entropy": 0.5,
                "linguistic": 0.5,
                "structure": 0.5,
                "diversity": 0.5
            }
        
        return analysis_results
    
    def _calculate_perplexity_score(self, text: str) -> float:
        """
        Calculate perplexity-based AI detection score.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            float: Perplexity-based detection score (0.0-1.0)
        """
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_sequence_length,
                padding=False
            )
            
            # Calculate perplexity
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss.item()
                perplexity = math.exp(loss)
            
            # Convert perplexity to detection score
            # Lower perplexity suggests more predictable (AI-generated) text
            if perplexity <= self.perplexity_threshold:
                # High confidence of AI generation
                score = 1.0 - (perplexity / self.perplexity_threshold) * 0.5
            else:
                # Lower confidence, likely human text
                score = max(0.0, 0.5 - (perplexity - self.perplexity_threshold) / 50.0)
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")
            return 0.5
    
    def _analyze_repetition_patterns(self, words: List[str]) -> float:
        """
        Analyze repetition patterns in the text.
        
        AI-generated text often shows characteristic repetition patterns
        that differ from human writing.
        
        Args:
            words (List[str]): List of words to analyze
            
        Returns:
            float: Repetition-based detection score (0.0-1.0)
        """
        try:
            if len(words) < 5:
                return 0.5
            
            # Count immediate word repetitions
            immediate_reps = sum(1 for i in range(len(words) - 1) 
                               if words[i].lower() == words[i + 1].lower())
            immediate_ratio = immediate_reps / max(1, len(words) - 1)
            
            # Count phrase repetitions (2-3 word sequences)
            bigram_reps = self._count_ngram_repetitions(words, 2)
            trigram_reps = self._count_ngram_repetitions(words, 3)
            
            # Calculate overall repetition score
            repetition_score = (immediate_ratio * 0.4 + bigram_reps * 0.4 + trigram_reps * 0.2)
            
            # Normalize and convert to AI detection confidence
            normalized_score = min(1.0, repetition_score / self.repetition_threshold)
            
            return normalized_score
            
        except Exception as e:
            logger.warning(f"Repetition analysis failed: {e}")
            return 0.5
    
    def _count_ngram_repetitions(self, words: List[str], n: int) -> float:
        """
        Count n-gram repetitions in the word sequence.
        
        Args:
            words (List[str]): List of words
            n (int): N-gram size
            
        Returns:
            float: Repetition ratio for n-grams
        """
        if len(words) < n:
            return 0.0
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(word.lower() for word in words[i:i + n])
            ngrams.append(ngram)
        
        if not ngrams:
            return 0.0
        
        # Count repetitions
        ngram_counts = Counter(ngrams)
        repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
        
        return repeated_ngrams / len(set(ngrams))
    
    def _analyze_burstiness(self, words: List[str]) -> float:
        """
        Analyze burstiness patterns in word usage.
        
        Burstiness measures how "bursty" or clustered word usage is.
        AI text often has lower burstiness than human text.
        
        Args:
            words (List[str]): List of words to analyze
            
        Returns:
            float: Burstiness-based detection score (0.0-1.0)
        """
        try:
            if len(words) < 10:
                return 0.5
            
            # Count word frequencies
            word_counts = Counter(word.lower() for word in words)
            
            # Calculate burstiness for frequent words
            burstiness_scores = []
            
            for word, count in word_counts.items():
                if count >= 3:  # Only analyze words that appear multiple times
                    # Find positions of word occurrences
                    positions = [i for i, w in enumerate(words) if w.lower() == word]
                    
                    if len(positions) > 1:
                        # Calculate inter-arrival times
                        intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                        
                        if len(intervals) > 1:
                            # Calculate burstiness (coefficient of variation)
                            mean_interval = sum(intervals) / len(intervals)
                            variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
                            
                            if mean_interval > 0:
                                burstiness = math.sqrt(variance) / mean_interval
                                burstiness_scores.append(burstiness)
            
            if not burstiness_scores:
                return 0.5
            
            # Average burstiness
            avg_burstiness = sum(burstiness_scores) / len(burstiness_scores)
            
            # Convert to AI detection score (lower burstiness suggests AI)
            if avg_burstiness <= self.burstiness_threshold:
                score = 1.0 - (avg_burstiness / self.burstiness_threshold) * 0.5
            else:
                score = max(0.0, 0.5 - (avg_burstiness - self.burstiness_threshold) / 2.0)
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.warning(f"Burstiness analysis failed: {e}")
            return 0.5
    
    def _analyze_entropy(self, words: List[str]) -> float:
        """
        Analyze entropy patterns in word distribution.
        
        Args:
            words (List[str]): List of words to analyze
            
        Returns:
            float: Entropy-based detection score (0.0-1.0)
        """
        try:
            if len(words) < 5:
                return 0.5
            
            # Calculate word frequency distribution
            word_counts = Counter(word.lower() for word in words)
            total_words = len(words)
            
            # Calculate Shannon entropy
            entropy = 0.0
            for count in word_counts.values():
                probability = count / total_words
                if probability > 0:
                    entropy -= probability * math.log2(probability)
            
            # Normalize entropy (typical range: 1-8 for natural text)
            normalized_entropy = entropy / 8.0
            
            # Convert to AI detection score
            # AI text often has different entropy patterns
            if entropy <= self.entropy_threshold:
                score = 0.7 + (self.entropy_threshold - entropy) / self.entropy_threshold * 0.3
            else:
                score = max(0.0, 0.7 - (entropy - self.entropy_threshold) / 4.0)
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.warning(f"Entropy analysis failed: {e}")
            return 0.5  
  
    def _analyze_linguistic_features(self, text: str, words: List[str]) -> float:
        """
        Analyze linguistic features characteristic of AI-generated text.
        
        Args:
            text (str): Original text
            words (List[str]): List of words
            
        Returns:
            float: Linguistic feature-based detection score (0.0-1.0)
        """
        try:
            features_score = 0.0
            feature_count = 0
            
            # Average word length (AI often uses more consistent word lengths)
            avg_word_length = sum(len(word) for word in words) / len(words)
            if 4.0 <= avg_word_length <= 6.0:  # Typical AI range
                features_score += 0.7
            else:
                features_score += 0.3
            feature_count += 1
            
            # Sentence length variation
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) > 1:
                sentence_lengths = [len(s.split()) for s in sentences]
                avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
                
                # Calculate coefficient of variation
                if avg_sentence_length > 0:
                    variance = sum((x - avg_sentence_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
                    cv = math.sqrt(variance) / avg_sentence_length
                    
                    # AI text often has lower sentence length variation
                    if cv < 0.5:
                        features_score += 0.8
                    else:
                        features_score += 0.2
                    feature_count += 1
            
            # Punctuation patterns
            punctuation_count = len(re.findall(r'[.!?,:;]', text))
            punctuation_ratio = punctuation_count / len(words)
            
            # AI text often has consistent punctuation usage
            if 0.05 <= punctuation_ratio <= 0.15:
                features_score += 0.6
            else:
                features_score += 0.4
            feature_count += 1
            
            # Function word ratio (the, and, of, etc.)
            function_words = {
                'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that',
                'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they',
                'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had',
                'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when'
            }
            
            function_word_count = sum(1 for word in words if word.lower() in function_words)
            function_word_ratio = function_word_count / len(words)
            
            # AI text often has consistent function word usage
            if 0.3 <= function_word_ratio <= 0.5:
                features_score += 0.7
            else:
                features_score += 0.3
            feature_count += 1
            
            return features_score / max(1, feature_count) if feature_count > 0 else 0.5
            
        except Exception as e:
            logger.warning(f"Linguistic feature analysis failed: {e}")
            return 0.5
    
    def _analyze_sentence_structure(self, text: str) -> float:
        """
        Analyze sentence structure patterns.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            float: Structure-based detection score (0.0-1.0)
        """
        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return 0.5
            
            structure_score = 0.0
            
            # Analyze sentence beginnings
            beginnings = []
            for sentence in sentences:
                words = sentence.split()
                if words:
                    beginnings.append(words[0].lower())
            
            # Check for repetitive sentence beginnings (common in AI text)
            beginning_counts = Counter(beginnings)
            max_beginning_count = max(beginning_counts.values()) if beginning_counts else 1
            beginning_repetition = max_beginning_count / len(beginnings)
            
            if beginning_repetition > 0.3:  # High repetition suggests AI
                structure_score += 0.8
            else:
                structure_score += 0.2
            
            # Analyze sentence complexity (simple heuristic)
            complex_sentences = 0
            for sentence in sentences:
                # Count subordinate clauses and conjunctions
                complexity_markers = len(re.findall(r'\b(because|although|since|while|if|when|that|which|who)\b', sentence.lower()))
                if complexity_markers >= 2:
                    complex_sentences += 1
            
            complexity_ratio = complex_sentences / len(sentences)
            
            # AI text often has moderate complexity
            if 0.2 <= complexity_ratio <= 0.6:
                structure_score += 0.6
            else:
                structure_score += 0.4
            
            return structure_score / 2.0  # Average of the two measures
            
        except Exception as e:
            logger.warning(f"Sentence structure analysis failed: {e}")
            return 0.5
    
    def _analyze_vocabulary_diversity(self, words: List[str]) -> float:
        """
        Analyze vocabulary diversity patterns.
        
        Args:
            words (List[str]): List of words to analyze
            
        Returns:
            float: Diversity-based detection score (0.0-1.0)
        """
        try:
            if len(words) < 10:
                return 0.5
            
            # Calculate type-token ratio (TTR)
            unique_words = len(set(word.lower() for word in words))
            ttr = unique_words / len(words)
            
            # Calculate moving average TTR for longer texts
            if len(words) > 100:
                # Calculate TTR for 50-word windows
                window_ttrs = []
                window_size = 50
                
                for i in range(0, len(words) - window_size + 1, window_size // 2):
                    window_words = words[i:i + window_size]
                    window_unique = len(set(word.lower() for word in window_words))
                    window_ttr = window_unique / len(window_words)
                    window_ttrs.append(window_ttr)
                
                if window_ttrs:
                    avg_ttr = sum(window_ttrs) / len(window_ttrs)
                    ttr_variance = sum((x - avg_ttr) ** 2 for x in window_ttrs) / len(window_ttrs)
                    ttr_consistency = 1.0 - min(1.0, math.sqrt(ttr_variance) * 10)
                else:
                    ttr_consistency = 0.5
            else:
                ttr_consistency = 0.5
            
            # AI text often has consistent vocabulary diversity
            diversity_score = 0.0
            
            # TTR analysis (AI often has moderate TTR)
            if 0.4 <= ttr <= 0.8:
                diversity_score += 0.7
            else:
                diversity_score += 0.3
            
            # TTR consistency (AI often more consistent)
            if ttr_consistency > 0.6:
                diversity_score += 0.8
            else:
                diversity_score += 0.2
            
            return diversity_score / 2.0
            
        except Exception as e:
            logger.warning(f"Vocabulary diversity analysis failed: {e}")
            return 0.5
    
    def _calculate_confidence_score(self, analysis_results: Dict[str, float]) -> float:
        """
        Calculate overall confidence score from statistical analysis results.
        
        Args:
            analysis_results (Dict[str, float]): Individual analysis scores
            
        Returns:
            float: Overall confidence score (0.0-1.0)
        """
        # Weighted combination of analysis results
        weights = {
            "perplexity": 0.25,      # High weight for perplexity if available
            "repetition": 0.20,      # Important for AI detection
            "burstiness": 0.15,      # Characteristic AI pattern
            "entropy": 0.15,         # Statistical measure
            "linguistic": 0.10,      # Linguistic features
            "structure": 0.10,       # Sentence structure
            "diversity": 0.05        # Vocabulary diversity
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, score in analysis_results.items():
            weight = weights.get(metric, 0.05)  # Default small weight for unknown metrics
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        final_score = weighted_score / total_weight
        return min(1.0, max(0.0, final_score))
    
    def _compile_detection_metadata(
        self,
        text: str,
        words: List[str],
        analysis_results: Dict[str, float],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Compile comprehensive metadata about the detection process.
        
        Args:
            text (str): Original input text
            words (List[str]): List of words
            analysis_results (Dict[str, float]): Analysis results
            start_time (float): Detection start timestamp
            
        Returns:
            Dict[str, Any]: Comprehensive detection metadata
        """
        return {
            "detector_info": {
                "detector_name": self.name,
                "reference_model": self.model_name,
                "model_loaded": self.model is not None,
                "detection_count": self.detection_count
            },
            "text_analysis": {
                "text_length": len(text),
                "word_count": len(words),
                "sentence_count": len(re.split(r'[.!?]+', text)),
                "avg_word_length": sum(len(word) for word in words) / len(words),
                "unique_words": len(set(word.lower() for word in words))
            },
            "statistical_scores": analysis_results,
            "detection_thresholds": {
                "perplexity_threshold": self.perplexity_threshold,
                "repetition_threshold": self.repetition_threshold,
                "burstiness_threshold": self.burstiness_threshold,
                "entropy_threshold": self.entropy_threshold
            },
            "analysis_methods": {
                "perplexity_available": self.model is not None,
                "statistical_analysis": True,
                "linguistic_features": True,
                "structure_analysis": True
            }
        }
    
    def is_available(self) -> bool:
        """
        Check if the custom detector is ready for use.
        
        The custom detector is always available as it doesn't require
        external models for basic statistical analysis.
        
        Returns:
            bool: True (custom detector is always available)
        """
        try:
            # Try to initialize if not already done
            if not self._is_initialized:
                self._lazy_load_model()
            
            # Custom detector is always available for statistical analysis
            return True
            
        except Exception as e:
            logger.warning(f"Custom detector availability check failed: {e}")
            # Even if model loading fails, statistical methods still work
            return True
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of models this detector can work with.
        
        Returns:
            List[str]: List of supported model identifiers
        """
        return [
            "microsoft/Phi-3-mini-4k-instruct",
            "microsoft/Phi-3-small-8k-instruct",
            "microsoft/Phi-3-medium-14k-instruct",
            "ai-generated"  # Generic AI detection
        ]
    
    def cleanup(self) -> None:
        """
        Clean up detector resources and free memory.
        """
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Custom detector resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Custom detector cleanup failed: {e}")
    
    def get_detection_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the detector capabilities.
        
        Returns:
            Dict[str, Any]: Detector information and statistics
        """
        return {
            "detector_name": self.name,
            "reference_model": self.model_name,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "detection_count": self.detection_count,
            "capabilities": {
                "perplexity_analysis": self.model is not None,
                "statistical_analysis": True,
                "linguistic_features": True,
                "structure_analysis": True,
                "vocabulary_analysis": True
            },
            "thresholds": {
                "perplexity": self.perplexity_threshold,
                "repetition": self.repetition_threshold,
                "burstiness": self.burstiness_threshold,
                "entropy": self.entropy_threshold
            },
            "supported_models": self.get_supported_models()
        }
    
    def update_thresholds(self, **kwargs) -> None:
        """
        Update detection thresholds for fine-tuning.
        
        Args:
            **kwargs: Threshold parameters to update
        """
        if "perplexity_threshold" in kwargs:
            self.perplexity_threshold = kwargs["perplexity_threshold"]
        
        if "repetition_threshold" in kwargs:
            self.repetition_threshold = kwargs["repetition_threshold"]
        
        if "burstiness_threshold" in kwargs:
            self.burstiness_threshold = kwargs["burstiness_threshold"]
        
        if "entropy_threshold" in kwargs:
            self.entropy_threshold = kwargs["entropy_threshold"]
        
        logger.info(f"Updated custom detector thresholds: {kwargs}")


def create_custom_detector(model_name: Optional[str] = None) -> CustomDetector:
    """
    Create a CustomDetector instance with optional model specification.
    
    Args:
        model_name (Optional[str]): Reference model for perplexity analysis
        
    Returns:
        CustomDetector: Configured custom detector instance
    """
    return CustomDetector(model_name)


def create_statistical_detector() -> CustomDetector:
    """
    Create a CustomDetector that relies only on statistical analysis.
    
    This function creates a detector that doesn't load any models and
    relies purely on statistical and linguistic analysis methods.
    
    Returns:
        CustomDetector: Statistical-only detector instance
    """
    detector = CustomDetector()
    # Prevent model loading by setting as initialized
    detector._is_initialized = True
    detector.model = None
    detector.tokenizer = None
    
    return detector