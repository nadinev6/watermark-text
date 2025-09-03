"""
Watermarking service for embedding and extracting watermarks in text.

This module provides the core watermarking functionality using the stegano library
for steganographic text watermarking. It supports both hidden (LSB) and visible
watermarking methods with comprehensive error handling and validation.
"""

import time
import logging
import hashlib
import re
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from stegano import lsb_text
from models.schemas import (
    WatermarkEmbedRequest,
    WatermarkEmbedResponse,
    WatermarkExtractRequest,
    WatermarkExtractResponse,
    WatermarkValidationRequest,
    WatermarkValidationResponse,
    WatermarkMethod,
    WatermarkVisibility
)
from utils.config import get_config

logger = logging.getLogger(__name__)


class WatermarkingError(Exception):
    """
    Custom exception for watermarking operations.
    
    This exception is raised when watermarking operations fail due to
    invalid input, processing errors, or stegano library issues.
    
    Attributes:
        message (str): Human-readable error description
        error_code (str): Machine-readable error identifier
        recoverable (bool): Whether the error can be recovered from
    """
    
    def __init__(self, message: str, error_code: str, recoverable: bool = True):
        """
        Initialize watermarking error.
        
        Args:
            message (str): Descriptive error message
            error_code (str): Unique error code for programmatic handling
            recoverable (bool): Whether operation can be retried
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.recoverable = recoverable


class WatermarkService:
    """
    Service class for text watermarking operations.
    
    This class provides functionality to embed and extract watermarks in text
    using steganographic techniques. It supports multiple watermarking methods
    and provides comprehensive validation and error handling.
    """
    
    def __init__(self):
        """Initialize the watermarking service."""
        self.config = get_config()
        
        # Watermarking statistics
        self.total_embeds = 0
        self.successful_embeds = 0
        self.failed_embeds = 0
        self.total_extractions = 0
        self.successful_extractions = 0
        self.failed_extractions = 0
        
        # Watermarking constraints
        self.min_text_length = 50  # Minimum text length for reliable watermarking
        self.max_text_length = getattr(self.config, 'MAX_TEXT_LENGTH', 50000)
        self.max_watermark_length = 500  # Maximum watermark content length
        
        logger.info("Watermarking service initialized")
    
    async def embed_watermark(self, request: WatermarkEmbedRequest) -> WatermarkEmbedResponse:
        """
        Embed a watermark into text using the specified method.
        
        This method takes input text and watermark content, then embeds the
        watermark using the chosen steganographic technique. The watermarked
        text appears identical to the original but contains hidden information.
        
        Args:
            request (WatermarkEmbedRequest): Embedding request with text and watermark
            
        Returns:
            WatermarkEmbedResponse: Response with watermarked text and metadata
            
        Raises:
            WatermarkingError: If embedding fails or invalid parameters
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_embed_request(request)
            
            # Perform watermarking based on method
            if request.method == WatermarkMethod.STEGANO_LSB:
                watermarked_text, metadata = await self._embed_stegano_watermark(
                    request.text, request.watermark_content, request.visibility
                )
            elif request.method == WatermarkMethod.VISIBLE_TEXT:
                watermarked_text, metadata = await self._embed_visible_watermark(
                    request.text, request.watermark_content, request.visibility
                )
            else:
                raise WatermarkingError(
                    f"Unsupported watermarking method: {request.method}",
                    "UNSUPPORTED_METHOD",
                    recoverable=False
                )
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            # Generate watermark hash for verification
            watermark_hash = self._generate_watermark_hash(
                request.text, request.watermark_content, request.method.value
            )
            
            # Compile embedding metadata
            embed_metadata = self._compile_embed_metadata(
                request, watermarked_text, metadata, processing_time
            )
            
            # Create response
            response = WatermarkEmbedResponse(
                watermarked_text=watermarked_text,
                watermark_hash=watermark_hash,
                method_used=request.method,
                visibility=request.visibility,
                processing_time_ms=processing_time,
                metadata=embed_metadata,
                warnings=metadata.get("warnings", [])
            )
            
            # Update statistics
            self.total_embeds += 1
            self.successful_embeds += 1
            
            logger.info(
                f"Watermark embedded successfully: method={request.method.value}, "
                f"visibility={request.visibility.value}, time={processing_time}ms"
            )
            
            return response
            
        except WatermarkingError:
            self.failed_embeds += 1
            raise
        except Exception as e:
            self.failed_embeds += 1
            logger.error(f"Watermark embedding failed: {e}")
            raise WatermarkingError(
                f"Embedding operation failed: {str(e)}",
                "EMBEDDING_FAILED",
                recoverable=True
            )
    
    async def extract_watermark(self, request: WatermarkExtractRequest) -> WatermarkExtractResponse:
        """
        Extract a watermark from watermarked text.
        
        This method analyzes text to extract any embedded watermarks using
        the specified detection methods. It can detect both hidden and visible
        watermarks depending on the methods used.
        
        Args:
            request (WatermarkExtractRequest): Extraction request with text to analyze
            
        Returns:
            WatermarkExtractResponse: Response with extracted watermark and metadata
            
        Raises:
            WatermarkingError: If extraction fails or invalid parameters
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_extract_request(request)
            
            # Attempt extraction with different methods
            extraction_results = []
            
            # Try stegano extraction if requested
            if WatermarkMethod.STEGANO_LSB in request.methods:
                stegano_result = await self._extract_stegano_watermark(request.text)
                if stegano_result:
                    extraction_results.append(stegano_result)
            
            # Try visible watermark detection if requested
            if WatermarkMethod.VISIBLE_TEXT in request.methods:
                visible_result = await self._extract_visible_watermark(request.text)
                if visible_result:
                    extraction_results.append(visible_result)
            
            # Determine best extraction result
            best_result = self._select_best_extraction(extraction_results)
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            # Compile extraction metadata
            extract_metadata = self._compile_extract_metadata(
                request, extraction_results, processing_time
            )
            
            # Create response
            response = WatermarkExtractResponse(
                watermark_found=best_result is not None,
                watermark_content=best_result["content"] if best_result else None,
                extraction_method=best_result["method"] if best_result else None,
                confidence_score=best_result["confidence"] if best_result else 0.0,
                processing_time_ms=processing_time,
                metadata=extract_metadata,
                warnings=extract_metadata.get("warnings", [])
            )
            
            # Update statistics
            self.total_extractions += 1
            if best_result:
                self.successful_extractions += 1
            
            logger.info(
                f"Watermark extraction completed: found={response.watermark_found}, "
                f"method={response.extraction_method}, time={processing_time}ms"
            )
            
            return response
            
        except WatermarkingError:
            self.failed_extractions += 1
            raise
        except Exception as e:
            self.failed_extractions += 1
            logger.error(f"Watermark extraction failed: {e}")
            raise WatermarkingError(
                f"Extraction operation failed: {str(e)}",
                "EXTRACTION_FAILED",
                recoverable=True
            )
    
    async def validate_watermark(self, request: WatermarkValidationRequest) -> WatermarkValidationResponse:
        """
        Validate the integrity of a watermarked text.
        
        This method checks if a watermarked text still contains its original
        watermark and validates the integrity of the embedding.
        
        Args:
            request (WatermarkValidationRequest): Validation request parameters
            
        Returns:
            WatermarkValidationResponse: Validation results and integrity status
        """
        start_time = time.time()
        
        try:
            # Extract watermark from the text
            extract_request = WatermarkExtractRequest(
                text=request.watermarked_text,
                methods=[WatermarkMethod.STEGANO_LSB, WatermarkMethod.VISIBLE_TEXT]
            )
            
            extraction_result = await self.extract_watermark(extract_request)
            
            # Validate against expected watermark
            is_valid = False
            integrity_score = 0.0
            
            if extraction_result.watermark_found and request.expected_watermark:
                extracted_content = extraction_result.watermark_content or ""
                is_valid = extracted_content == request.expected_watermark
                
                # Calculate integrity score based on exact match and confidence
                if is_valid:
                    integrity_score = extraction_result.confidence_score
                else:
                    # Partial match scoring
                    similarity = self._calculate_text_similarity(
                        extracted_content, request.expected_watermark
                    )
                    integrity_score = similarity * extraction_result.confidence_score
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            # Compile validation metadata
            validation_metadata = {
                "validation_method": "extraction_comparison",
                "expected_watermark_length": len(request.expected_watermark) if request.expected_watermark else 0,
                "extracted_watermark_length": len(extraction_result.watermark_content or ""),
                "extraction_confidence": extraction_result.confidence_score,
                "extraction_method": extraction_result.extraction_method,
                "text_analysis": {
                    "text_length": len(request.watermarked_text),
                    "word_count": len(request.watermarked_text.split()),
                    "character_count": len(request.watermarked_text)
                }
            }
            
            # Create response
            response = WatermarkValidationResponse(
                is_valid=is_valid,
                integrity_score=integrity_score,
                extracted_watermark=extraction_result.watermark_content,
                validation_method="extraction_comparison",
                processing_time_ms=processing_time,
                metadata=validation_metadata,
                warnings=[]
            )
            
            logger.info(
                f"Watermark validation completed: valid={is_valid}, "
                f"integrity={integrity_score:.3f}, time={processing_time}ms"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Watermark validation failed: {e}")
            raise WatermarkingError(
                f"Validation operation failed: {str(e)}",
                "VALIDATION_FAILED",
                recoverable=True
            )
    
    def _validate_embed_request(self, request: WatermarkEmbedRequest) -> None:
        """
        Validate watermark embedding request parameters.
        
        Args:
            request (WatermarkEmbedRequest): Request to validate
            
        Raises:
            WatermarkingError: If validation fails
        """
        # Check text length
        if len(request.text) < self.min_text_length:
            raise WatermarkingError(
                f"Text too short for watermarking: minimum {self.min_text_length} characters",
                "TEXT_TOO_SHORT",
                recoverable=False
            )
        
        if len(request.text) > self.max_text_length:
            raise WatermarkingError(
                f"Text too long for watermarking: maximum {self.max_text_length} characters",
                "TEXT_TOO_LONG",
                recoverable=False
            )
        
        # Check watermark content
        if not request.watermark_content.strip():
            raise WatermarkingError(
                "Watermark content cannot be empty",
                "EMPTY_WATERMARK",
                recoverable=False
            )
        
        if len(request.watermark_content) > self.max_watermark_length:
            raise WatermarkingError(
                f"Watermark too long: maximum {self.max_watermark_length} characters",
                "WATERMARK_TOO_LONG",
                recoverable=False
            )
        
        # Validate ASCII content for stegano method
        if request.method == WatermarkMethod.STEGANO_LSB:
            try:
                request.watermark_content.encode('ascii')
            except UnicodeEncodeError:
                raise WatermarkingError(
                    "Watermark content must be ASCII for stegano method",
                    "NON_ASCII_WATERMARK",
                    recoverable=False
                )
    
    def _validate_extract_request(self, request: WatermarkExtractRequest) -> None:
        """
        Validate watermark extraction request parameters.
        
        Args:
            request (WatermarkExtractRequest): Request to validate
            
        Raises:
            WatermarkingError: If validation fails
        """
        # Check text length
        if len(request.text) < 10:  # Minimum for extraction
            raise WatermarkingError(
                "Text too short for watermark extraction",
                "TEXT_TOO_SHORT",
                recoverable=False
            )
        
        if len(request.text) > self.max_text_length:
            raise WatermarkingError(
                f"Text too long for extraction: maximum {self.max_text_length} characters",
                "TEXT_TOO_LONG",
                recoverable=False
            )
        
        # Check methods
        if not request.methods:
            raise WatermarkingError(
                "At least one extraction method must be specified",
                "NO_METHODS_SPECIFIED",
                recoverable=False
            )
    
    async def _embed_stegano_watermark(
        self,
        text: str,
        watermark_content: str,
        visibility: WatermarkVisibility
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Embed watermark using stegano LSB method.
        
        Args:
            text (str): Original text
            watermark_content (str): Watermark to embed
            visibility (WatermarkVisibility): Visibility setting
            
        Returns:
            Tuple[str, Dict[str, Any]]: Watermarked text and metadata
        """
        try:
            # Use stegano to hide the watermark in the text
            watermarked_text = lsb_text.hide(text, watermark_content)
            
            # Calculate embedding statistics
            original_size = len(text.encode('utf-8'))
            watermarked_size = len(watermarked_text.encode('utf-8'))
            size_increase = watermarked_size - original_size
            
            metadata = {
                "embedding_method": "stegano_lsb",
                "original_length": len(text),
                "watermarked_length": len(watermarked_text),
                "watermark_length": len(watermark_content),
                "size_increase_bytes": size_increase,
                "size_increase_percent": (size_increase / original_size) * 100 if original_size > 0 else 0,
                "character_changes": self._count_character_changes(text, watermarked_text),
                "warnings": []
            }
            
            # Add warnings if significant changes
            if size_increase > 1000:  # More than 1KB increase
                metadata["warnings"].append("Significant size increase detected")
            
            if metadata["character_changes"] > len(text) * 0.1:  # More than 10% changes
                metadata["warnings"].append("High number of character modifications")
            
            logger.info(f"Stegano watermark embedded: {len(watermark_content)} chars into {len(text)} chars")
            
            return watermarked_text, metadata
            
        except Exception as e:
            logger.error(f"Stegano embedding failed: {e}")
            raise WatermarkingError(
                f"Stegano watermark embedding failed: {str(e)}",
                "STEGANO_EMBED_FAILED",
                recoverable=True
            )
    
    async def _embed_visible_watermark(
        self,
        text: str,
        watermark_content: str,
        visibility: WatermarkVisibility
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Embed visible watermark into text.
        
        Args:
            text (str): Original text
            watermark_content (str): Watermark to embed
            visibility (WatermarkVisibility): Visibility setting
            
        Returns:
            Tuple[str, Dict[str, Any]]: Watermarked text and metadata
        """
        try:
            # Format watermark based on visibility
            if visibility == WatermarkVisibility.HIDDEN:
                # For "hidden" visible watermarks, use subtle formatting
                formatted_watermark = f"\n\n<!-- {watermark_content} -->"
            else:
                # For visible watermarks, add clear attribution
                formatted_watermark = f"\n\n[Watermark: {watermark_content}]"
            
            # Append watermark to text
            watermarked_text = text + formatted_watermark
            
            metadata = {
                "embedding_method": "visible_text",
                "original_length": len(text),
                "watermarked_length": len(watermarked_text),
                "watermark_length": len(watermark_content),
                "watermark_position": "end",
                "watermark_format": "html_comment" if visibility == WatermarkVisibility.HIDDEN else "bracketed_text",
                "warnings": []
            }
            
            logger.info(f"Visible watermark embedded: {len(watermark_content)} chars")
            
            return watermarked_text, metadata
            
        except Exception as e:
            logger.error(f"Visible embedding failed: {e}")
            raise WatermarkingError(
                f"Visible watermark embedding failed: {str(e)}",
                "VISIBLE_EMBED_FAILED",
                recoverable=True
            )
    
    async def _extract_stegano_watermark(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract watermark using stegano LSB method.
        
        Args:
            text (str): Text to extract watermark from
            
        Returns:
            Optional[Dict[str, Any]]: Extraction result or None if no watermark found
        """
        try:
            # Use stegano to reveal hidden watermark
            extracted_content = lsb_text.reveal(text)
            
            if extracted_content:
                return {
                    "content": extracted_content,
                    "method": WatermarkMethod.STEGANO_LSB,
                    "confidence": 1.0,  # Stegano extraction is binary (found or not)
                    "metadata": {
                        "extraction_method": "stegano_lsb",
                        "extracted_length": len(extracted_content),
                        "source_text_length": len(text)
                    }
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Stegano extraction failed: {e}")
            return None
    
    async def _extract_visible_watermark(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract visible watermark from text.
        
        Args:
            text (str): Text to extract watermark from
            
        Returns:
            Optional[Dict[str, Any]]: Extraction result or None if no watermark found
        """
        try:
            # Look for HTML comment watermarks
            html_pattern = r'<!--\s*(.+?)\s*-->'
            html_matches = re.findall(html_pattern, text, re.DOTALL)
            
            # Look for bracketed watermarks
            bracket_pattern = r'\[Watermark:\s*(.+?)\]'
            bracket_matches = re.findall(bracket_pattern, text, re.DOTALL)
            
            # Combine all matches
            all_matches = []
            
            for match in html_matches:
                all_matches.append({
                    "content": match.strip(),
                    "format": "html_comment",
                    "confidence": 0.9
                })
            
            for match in bracket_matches:
                all_matches.append({
                    "content": match.strip(),
                    "format": "bracketed_text",
                    "confidence": 1.0
                })
            
            if all_matches:
                # Return the match with highest confidence
                best_match = max(all_matches, key=lambda x: x["confidence"])
                
                return {
                    "content": best_match["content"],
                    "method": WatermarkMethod.VISIBLE_TEXT,
                    "confidence": best_match["confidence"],
                    "metadata": {
                        "extraction_method": "visible_text",
                        "watermark_format": best_match["format"],
                        "total_matches_found": len(all_matches),
                        "extracted_length": len(best_match["content"])
                    }
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Visible extraction failed: {e}")
            return None
    
    def _select_best_extraction(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Select the best extraction result from multiple methods.
        
        Args:
            results (List[Dict[str, Any]]): List of extraction results
            
        Returns:
            Optional[Dict[str, Any]]: Best extraction result or None
        """
        if not results:
            return None
        
        # Sort by confidence score (highest first)
        sorted_results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        
        return sorted_results[0]
    
    def _generate_watermark_hash(self, text: str, watermark: str, method: str) -> str:
        """
        Generate a hash for watermark verification.
        
        Args:
            text (str): Original text
            watermark (str): Watermark content
            method (str): Watermarking method
            
        Returns:
            str: SHA256 hash for verification
        """
        # Create a unique identifier for this watermarking operation
        content = f"{text}|{watermark}|{method}|{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _count_character_changes(self, original: str, watermarked: str) -> int:
        """
        Count the number of character changes between original and watermarked text.
        
        Args:
            original (str): Original text
            watermarked (str): Watermarked text
            
        Returns:
            int: Number of character differences
        """
        if len(original) != len(watermarked):
            return abs(len(original) - len(watermarked))
        
        changes = sum(1 for a, b in zip(original, watermarked) if a != b)
        return changes
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score (0.0-1.0)
        """
        if not text1 and not text2:
            return 1.0
        
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        max_len = max(len(text1), len(text2))
        min_len = min(len(text1), len(text2))
        
        # Calculate character matches
        matches = sum(1 for a, b in zip(text1, text2) if a == b)
        
        # Account for length differences
        similarity = matches / max_len
        
        return similarity
    
    def _compile_embed_metadata(
        self,
        request: WatermarkEmbedRequest,
        watermarked_text: str,
        method_metadata: Dict[str, Any],
        processing_time: int
    ) -> Dict[str, Any]:
        """
        Compile comprehensive metadata for embedding operation.
        
        Args:
            request (WatermarkEmbedRequest): Original request
            watermarked_text (str): Result watermarked text
            method_metadata (Dict[str, Any]): Method-specific metadata
            processing_time (int): Processing time in milliseconds
            
        Returns:
            Dict[str, Any]: Comprehensive embedding metadata
        """
        return {
            "operation": "watermark_embed",
            "request_info": {
                "method": request.method.value,
                "visibility": request.visibility.value,
                "preserve_formatting": request.preserve_formatting,
                "original_text_length": len(request.text),
                "watermark_content_length": len(request.watermark_content)
            },
            "result_info": {
                "watermarked_text_length": len(watermarked_text),
                "processing_time_ms": processing_time,
                "success": True
            },
            "method_specific": method_metadata,
            "service_stats": {
                "total_embeds": self.total_embeds + 1,
                "successful_embeds": self.successful_embeds + 1
            }
        }
    
    def _compile_extract_metadata(
        self,
        request: WatermarkExtractRequest,
        extraction_results: List[Dict[str, Any]],
        processing_time: int
    ) -> Dict[str, Any]:
        """
        Compile comprehensive metadata for extraction operation.
        
        Args:
            request (WatermarkExtractRequest): Original request
            extraction_results (List[Dict[str, Any]]): All extraction results
            processing_time (int): Processing time in milliseconds
            
        Returns:
            Dict[str, Any]: Comprehensive extraction metadata
        """
        return {
            "operation": "watermark_extract",
            "request_info": {
                "methods_attempted": [method.value for method in request.methods],
                "text_length": len(request.text)
            },
            "extraction_results": [
                {
                    "method": result["method"].value,
                    "confidence": result["confidence"],
                    "content_length": len(result["content"]),
                    "metadata": result["metadata"]
                }
                for result in extraction_results
            ],
            "result_info": {
                "total_methods_attempted": len(request.methods),
                "successful_extractions": len(extraction_results),
                "processing_time_ms": processing_time
            },
            "service_stats": {
                "total_extractions": self.total_extractions + 1,
                "successful_extractions": self.successful_extractions + len(extraction_results)
            }
        }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get watermarking service statistics.
        
        Returns:
            Dict[str, Any]: Service statistics and performance metrics
        """
        return {
            "embedding_stats": {
                "total_embeds": self.total_embeds,
                "successful_embeds": self.successful_embeds,
                "failed_embeds": self.failed_embeds,
                "success_rate": (
                    self.successful_embeds / max(1, self.total_embeds)
                )
            },
            "extraction_stats": {
                "total_extractions": self.total_extractions,
                "successful_extractions": self.successful_extractions,
                "failed_extractions": self.failed_extractions,
                "success_rate": (
                    self.successful_extractions / max(1, self.total_extractions)
                )
            },
            "capabilities": {
                "supported_methods": [method.value for method in WatermarkMethod],
                "supported_visibility": [vis.value for vis in WatermarkVisibility],
                "min_text_length": self.min_text_length,
                "max_text_length": self.max_text_length,
                "max_watermark_length": self.max_watermark_length
            },
            "service_info": {
                "stegano_available": True,  # Always available since it's a pure Python library
                "visible_watermarking_available": True
            }
        }
    
    def get_supported_methods(self) -> List[str]:
        """
        Get list of supported watermarking methods.
        
        Returns:
            List[str]: List of supported method identifiers
        """
        return [method.value for method in WatermarkMethod]
    
    def get_method_info(self, method: WatermarkMethod) -> Dict[str, Any]:
        """
        Get detailed information about a specific watermarking method.
        
        Args:
            method (WatermarkMethod): Method to get information about
            
        Returns:
            Dict[str, Any]: Method capabilities and limitations
        """
        if method == WatermarkMethod.STEGANO_LSB:
            return {
                "name": "Steganographic LSB",
                "description": "Hides watermark in least significant bits of characters",
                "visibility": "completely_hidden",
                "robustness": "high",
                "capacity": "medium",
                "requirements": {
                    "min_text_length": self.min_text_length,
                    "ascii_watermark": True,
                    "preserves_readability": True
                },
                "use_cases": [
                    "Copyright protection",
                    "Ownership verification", 
                    "Document authentication",
                    "E-book publishing"
                ]
            }
        elif method == WatermarkMethod.VISIBLE_TEXT:
            return {
                "name": "Visible Text Watermark",
                "description": "Adds visible watermark text to document",
                "visibility": "configurable",
                "robustness": "low",
                "capacity": "high",
                "requirements": {
                    "min_text_length": 10,
                    "ascii_watermark": False,
                    "preserves_readability": True
                },
                "use_cases": [
                    "Draft documents",
                    "Attribution notices",
                    "Copyright statements",
                    "Document versioning"
                ]
            }
        else:
            return {"error": "Unknown method"}


# Global watermarking service instance
_watermark_service: Optional[WatermarkService] = None


def get_watermark_service() -> WatermarkService:
    """
    Get or create the global watermarking service instance.
    
    Returns:
        WatermarkService: Global watermarking service
    """
    global _watermark_service
    if _watermark_service is None:
        _watermark_service = WatermarkService()
    return _watermark_service