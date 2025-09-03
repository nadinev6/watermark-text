"""
Watermarking API endpoints for text watermark operations.

This module provides FastAPI endpoints for watermark embedding, extraction,
and validation operations using steganographic and visible watermarking methods.
"""

import time
import logging
from typing import List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from models.schemas import (
    WatermarkEmbedRequest,
    WatermarkEmbedResponse,
    WatermarkExtractRequest, # Ensure this is imported
    WatermarkExtractResponse,
    WatermarkValidationRequest,
    WatermarkValidationResponse,
    WatermarkMethod,
    WatermarkVisibility
)
from services.watermark_service import (
    WatermarkService,
    WatermarkingError,
    get_watermark_service
)
from api.dependencies import validate_api_key, get_current_config

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/watermark", tags=["watermarking"])


@router.post("/embed", response_model=WatermarkEmbedResponse)
async def embed_watermark(
    request: WatermarkEmbedRequest,
    background_tasks: BackgroundTasks,
    service: WatermarkService = Depends(get_watermark_service),
    api_key: Optional[str] = Depends(validate_api_key)
) -> WatermarkEmbedResponse:
    """
    Embed a watermark into text using steganographic or visible methods.
    
    This endpoint takes input text and watermark content, then embeds the
    watermark using the specified method. The watermarked text maintains
    readability while containing the hidden or visible watermark.
    
    **Supported Methods:**
    - `stegano_lsb`: Hidden watermark using least significant bit steganography
    - `visible_text`: Visible watermark as text annotation
    
    **Use Cases:**
    - Copyright protection for documents
    - Author attribution for content
    - Document authentication and verification
    - E-book publishing with ownership tracking
    
    Args:
        request (WatermarkEmbedRequest): Embedding parameters including text,
                                       watermark content, method, and visibility
        background_tasks (BackgroundTasks): FastAPI background tasks
        service (WatermarkService): Watermarking service dependency
        api_key (Optional[str]): API key for authentication (if required)
        
    Returns:
        WatermarkEmbedResponse: Watermarked text with embedding metadata
        
    Raises:
        HTTPException: If embedding fails or invalid parameters provided
    """
    start_time = time.time()
    
    try:
        # --- NEW ANTI-PLAGIARISM CHECK ---
        # Create an extraction request for the current text
        extract_request = WatermarkExtractRequest(
            text=request.text,
            methods=[WatermarkMethod.STEGANO_LSB, WatermarkMethod.VISIBLE_TEXT] # Check both methods
        )
        
        # Attempt to extract any existing watermark
        existing_watermark_result = await service.extract_watermark(extract_request)
        
        # If a watermark is found, prevent embedding and raise an error
        if existing_watermark_result.watermark_found:
            raise HTTPException(
                status_code=409, # Conflict
                detail={
                    "error": "WATERMARK_OVERWRITE_DENIED",
                    "message": f"Overwrite denied: Text already contains a watermark ('{existing_watermark_result.watermark_content}'). Embedding a new watermark would corrupt the existing one.",
                    "recoverable": False
                }
            )
        # --- END NEW ANTI-PLAGIARISM CHECK ---

        logger.info(
            f"Watermark embed request: method={request.method.value}, "
            f"visibility={request.visibility.value}, text_length={len(request.text)}"
        )
        
        # Perform watermark embedding
        response = await service.embed_watermark(request)
        
        # Schedule background tasks for analytics (if needed)
        background_tasks.add_task(
            _log_watermark_operation,
            "embed",
            request.method.value,
            len(request.text),
            response.processing_time_ms
        )
        
        logger.info(
            f"Watermark embedded successfully: hash={response.watermark_hash[:16]}..., "
            f"time={response.processing_time_ms}ms"
        )
        
        return response
        
    except WatermarkingError as e:
        logger.warning(f"Watermarking error: {e.message} (code: {e.error_code})")
        
        # Map watermarking errors to appropriate HTTP status codes
        status_code = 400  # Default to bad request
        if e.error_code in ["TEXT_TOO_SHORT", "TEXT_TOO_LONG", "EMPTY_WATERMARK", "WATERMARK_TOO_LONG"]:
            status_code = 400  # Bad request
        elif e.error_code in ["UNSUPPORTED_METHOD"]:
            status_code = 422  # Unprocessable entity
        elif e.error_code in ["STEGANO_EMBED_FAILED", "VISIBLE_EMBED_FAILED"]:
            status_code = 500  # Internal server error
        
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "recoverable": e.recoverable
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error during watermark embedding: {e}")
        raise HTTPException(
            status_code=500,
            detail="Watermark embedding failed due to internal error"
        )

# ... (rest of the file) ...

    try:
        logger.info(
            f"Watermark embed request: method={request.method.value}, "
            f"visibility={request.visibility.value}, text_length={len(request.text)}"
        )
        
        # Perform watermark embedding
        response = await service.embed_watermark(request)
        
        # Schedule background tasks for analytics (if needed)
        background_tasks.add_task(
            _log_watermark_operation,
            "embed",
            request.method.value,
            len(request.text),
            response.processing_time_ms
        )
        
        logger.info(
            f"Watermark embedded successfully: hash={response.watermark_hash[:16]}..., "
            f"time={response.processing_time_ms}ms"
        )
        
        return response
        
    except WatermarkingError as e:
        logger.warning(f"Watermarking error: {e.message} (code: {e.error_code})")
        
        # Map watermarking errors to appropriate HTTP status codes
        status_code = 400  # Default to bad request
        if e.error_code in ["TEXT_TOO_SHORT", "TEXT_TOO_LONG", "EMPTY_WATERMARK", "WATERMARK_TOO_LONG"]:
            status_code = 400  # Bad request
        elif e.error_code in ["UNSUPPORTED_METHOD"]:
            status_code = 422  # Unprocessable entity
        elif e.error_code in ["STEGANO_EMBED_FAILED", "VISIBLE_EMBED_FAILED"]:
            status_code = 500  # Internal server error
        
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "recoverable": e.recoverable
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error during watermark embedding: {e}")
        raise HTTPException(
            status_code=500,
            detail="Watermark embedding failed due to internal error"
        )


@router.post("/extract", response_model=WatermarkExtractResponse)
async def extract_watermark(
    request: WatermarkExtractRequest,
    background_tasks: BackgroundTasks,
    service: WatermarkService = Depends(get_watermark_service),
    api_key: Optional[str] = Depends(validate_api_key)
) -> WatermarkExtractResponse:
    """
    Extract watermarks from watermarked text.
    
    This endpoint analyzes text to detect and extract any embedded watermarks
    using the specified extraction methods. It can detect both hidden
    steganographic watermarks and visible text watermarks.
    
    **Supported Methods:**
    - `stegano_lsb`: Extract hidden watermarks using LSB steganography
    - `visible_text`: Extract visible watermarks from text annotations
    
    **Detection Process:**
    1. Attempts extraction using all specified methods
    2. Returns the best result based on confidence scores
    3. Provides detailed metadata about the extraction process
    
    Args:
        request (WatermarkExtractRequest): Extraction parameters including text
                                         and methods to attempt
        background_tasks (BackgroundTasks): FastAPI background tasks
        service (WatermarkService): Watermarking service dependency
        api_key (Optional[str]): API key for authentication (if required)
        
    Returns:
        WatermarkExtractResponse: Extraction results with watermark content
                                and confidence information
        
    Raises:
        HTTPException: If extraction fails or invalid parameters provided
    """
    try:
        logger.info(
            f"Watermark extract request: methods={[m.value for m in request.methods]}, "
            f"text_length={len(request.text)}"
        )
        
        # Perform watermark extraction
        response = await service.extract_watermark(request)
        
        # Schedule background tasks for analytics
        background_tasks.add_task(
            _log_watermark_operation,
            "extract",
            ",".join(m.value for m in request.methods),
            len(request.text),
            response.processing_time_ms
        )
        
        logger.info(
            f"Watermark extraction completed: found={response.watermark_found}, "
            f"method={response.extraction_method}, confidence={response.confidence_score:.3f}"
        )
        
        return response
        
    except WatermarkingError as e:
        logger.warning(f"Watermarking error: {e.message} (code: {e.error_code})")
        
        # Map watermarking errors to appropriate HTTP status codes
        status_code = 400  # Default to bad request
        if e.error_code in ["TEXT_TOO_SHORT", "TEXT_TOO_LONG"]:
            status_code = 400  # Bad request
        elif e.error_code in ["NO_METHODS_SPECIFIED"]:
            status_code = 422  # Unprocessable entity
        elif e.error_code in ["EXTRACTION_FAILED"]:
            status_code = 500  # Internal server error
        
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "recoverable": e.recoverable
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error during watermark extraction: {e}")
        raise HTTPException(
            status_code=500,
            detail="Watermark extraction failed due to internal error"
        )


@router.post("/validate", response_model=WatermarkValidationResponse)
async def validate_watermark(
    request: WatermarkValidationRequest,
    background_tasks: BackgroundTasks,
    service: WatermarkService = Depends(get_watermark_service),
    api_key: Optional[str] = Depends(validate_api_key)
) -> WatermarkValidationResponse:
    """
    Validate the integrity of a watermarked text.
    
    This endpoint checks if a watermarked text still contains its original
    watermark and validates the integrity of the embedding. It's useful for
    verifying that watermarked content hasn't been tampered with.
    
    **Validation Process:**
    1. Extracts watermark from the provided text
    2. Compares extracted content with expected watermark
    3. Calculates integrity score based on match quality
    4. Provides detailed validation metadata
    
    Args:
        request (WatermarkValidationRequest): Validation parameters including
                                            watermarked text and expected watermark
        background_tasks (BackgroundTasks): FastAPI background tasks
        service (WatermarkService): Watermarking service dependency
        api_key (Optional[str]): API key for authentication (if required)
        
    Returns:
        WatermarkValidationResponse: Validation results with integrity score
                                   and extracted watermark content
        
    Raises:
        HTTPException: If validation fails or invalid parameters provided
    """
    try:
        logger.info(
            f"Watermark validation request: text_length={len(request.watermarked_text)}, "
            f"expected_watermark_length={len(request.expected_watermark or '')}"
        )
        
        # Perform watermark validation
        response = await service.validate_watermark(request)
        
        # Schedule background tasks for analytics
        background_tasks.add_task(
            _log_watermark_operation,
            "validate",
            "validation",
            len(request.watermarked_text),
            response.processing_time_ms
        )
        
        logger.info(
            f"Watermark validation completed: valid={response.is_valid}, "
            f"integrity={response.integrity_score:.3f}"
        )
        
        return response
        
    except WatermarkingError as e:
        logger.warning(f"Watermarking error: {e.message} (code: {e.error_code})")
        
        raise HTTPException(
            status_code=400,
            detail={
                "error": e.error_code,
                "message": e.message,
                "recoverable": e.recoverable
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error during watermark validation: {e}")
        raise HTTPException(
            status_code=500,
            detail="Watermark validation failed due to internal error"
        )


@router.get("/methods")
async def get_watermarking_methods(
    service: WatermarkService = Depends(get_watermark_service)
) -> dict:
    """
    Get information about available watermarking methods.
    
    This endpoint provides detailed information about all supported
    watermarking methods, their capabilities, and use cases.
    
    Args:
        service (WatermarkService): Watermarking service dependency
        
    Returns:
        dict: Available methods with detailed capability information
    """
    try:
        methods_info = {}
        
        for method in WatermarkMethod:
            method_info = service.get_method_info(method)
            methods_info[method.value] = method_info
        
        return {
            "supported_methods": service.get_supported_methods(),
            "method_details": methods_info,
            "visibility_options": [vis.value for vis in WatermarkVisibility],
            "service_capabilities": {
                "min_text_length": service.min_text_length,
                "max_text_length": service.max_text_length,
                "max_watermark_length": service.max_watermark_length
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get watermarking methods info: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve watermarking methods information"
        )


@router.get("/stats")
async def get_watermarking_stats(
    service: WatermarkService = Depends(get_watermark_service)
) -> dict:
    """
    Get watermarking service statistics and performance metrics.
    
    This endpoint provides comprehensive statistics about watermarking
    operations including success rates, performance metrics, and usage patterns.
    
    Args:
        service (WatermarkService): Watermarking service dependency
        
    Returns:
        dict: Service statistics and performance metrics
    """
    try:
        return service.get_service_stats()
        
    except Exception as e:
        logger.error(f"Failed to get watermarking stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve watermarking statistics"
        )


@router.get("/health")
async def watermarking_health_check(
    service: WatermarkService = Depends(get_watermark_service)
) -> dict:
    """
    Health check endpoint for watermarking service.
    
    This endpoint provides health status and availability information
    for the watermarking service and its dependencies.
    
    Args:
        service (WatermarkService): Watermarking service dependency
        
    Returns:
        dict: Health status and service availability information
    """
    try:
        stats = service.get_service_stats()
        
        return {
            "status": "healthy",
            "service_available": True,
            "stegano_available": stats["service_info"]["stegano_available"],
            "visible_watermarking_available": stats["service_info"]["visible_watermarking_available"],
            "supported_methods": stats["capabilities"]["supported_methods"],
            "total_operations": (
                stats["embedding_stats"]["total_embeds"] + 
                stats["extraction_stats"]["total_extractions"]
            ),
            "overall_success_rate": (
                (stats["embedding_stats"]["successful_embeds"] + 
                 stats["extraction_stats"]["successful_extractions"]) /
                max(1, stats["embedding_stats"]["total_embeds"] + 
                    stats["extraction_stats"]["total_extractions"])
            )
        }
        
    except Exception as e:
        logger.error(f"Watermarking health check failed: {e}")
        return {
            "status": "unhealthy",
            "service_available": False,
            "error": str(e)
        }


# Background task functions
async def _log_watermark_operation(
    operation_type: str,
    method: str,
    text_length: int,
    processing_time_ms: int
) -> None:
    """
    Log watermarking operation for analytics (background task).
    
    Args:
        operation_type (str): Type of operation (embed, extract, validate)
        method (str): Watermarking method used
        text_length (int): Length of processed text
        processing_time_ms (int): Processing time in milliseconds
    """
    try:
        logger.info(
            f"Watermark operation logged: type={operation_type}, "
            f"method={method}, text_length={text_length}, time={processing_time_ms}ms"
        )
        
        # In a production environment, you might want to:
        # - Store operation logs in database
        # - Send metrics to monitoring systems
        # - Update usage analytics
        
    except Exception as e:
        logger.error(f"Failed to log watermark operation: {e}")