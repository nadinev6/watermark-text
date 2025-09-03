"""
Detection API endpoints for watermark analysis.

This module provides FastAPI endpoints for watermark detection, including
single text analysis, batch processing, and detection result management.
"""

import time
import logging
from typing import List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
# api/detection.py
from stegano import text as stegano_text
from ..utils.config import get_config

async def detect_with_watermark(text: str):
    """Enhanced detection that checks for watermarks first"""
    config = get_config()
    
    # First check for watermarks
    try:
        watermark = stegano_text.reveal(text)
        if watermark:
            return {
                "has_watermark": True,
                "watermark_content": watermark,
                "detection_method": "stegano_extraction"
            }
    except:
        pass  # Continue with regular detection if watermark fails

from models.schemas import (
    DetectionRequest,
    DetectionResponse,
    BatchRequest,
    BatchResponse,
    HistoryResponse,
    AnalysisRecord
)
from detectors import (
    ResultCombiner,
    create_default_combiner,
    DetectionError
)

# Import mock detectors directly to avoid heavy model imports
try:
    from detectors.synthid_detector import SynthIDDetector
    from detectors.custom_detector import CustomDetector
except ImportError:
    # If real detectors fail to import, we'll use mocks
    SynthIDDetector = None
    CustomDetector = None
from utils.config import get_config
from api.dependencies import get_detection_service

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api", tags=["detection"])


class DetectionService:
    """
    Service class for managing watermark detection operations.
    
    This class coordinates between different detectors and provides
    a unified interface for watermark detection operations.
    """
    
    def __init__(self):
        """Initialize the detection service."""
        self.config = get_config()
        
        # Initialize detectors (lazy loading) - using Any to support mock detectors
        self._synthid_detector: Optional[Any] = None
        self._custom_detector: Optional[Any] = None
        self._result_combiner: Optional[ResultCombiner] = None
        
        # Detection statistics
        self.total_detections = 0
        self.successful_detections = 0
        self.failed_detections = 0
        
        logger.info("Detection service initialized")
    
    def _get_synthid_detector(self) -> Any:
        """Get or create SynthID detector instance."""
        if self._synthid_detector is None:
            # Always use mock detector for now to avoid model loading issues
            try:
                from detectors.mock_detector import MockSynthIDDetector
                self._synthid_detector = MockSynthIDDetector()
                logger.info("Mock SynthID detector initialized (default)")
            except Exception as e:
                logger.error(f"Failed to initialize mock SynthID detector: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="SynthID detector unavailable"
                )
        return self._synthid_detector
    
    def _get_custom_detector(self) -> Any:
        """Get or create custom detector instance."""
        if self._custom_detector is None:
            # Always use mock detector for now to avoid model loading issues
            try:
                from detectors.mock_detector import MockCustomDetector
                self._custom_detector = MockCustomDetector()
                logger.info("Mock custom detector initialized (default)")
            except Exception as e:
                logger.error(f"Failed to initialize mock custom detector: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="Custom detector unavailable"
                )
        return self._custom_detector
    
    def _get_result_combiner(self) -> ResultCombiner:
        """Get or create result combiner instance."""
        if self._result_combiner is None:
            self._result_combiner = create_default_combiner()
            logger.info("Result combiner initialized")
        return self._result_combiner
    
    async def detect_watermark(
        self,
        request: DetectionRequest,
        background_tasks: BackgroundTasks
    ) -> DetectionResponse:
        """
        Perform watermark detection on input text.
        
        Args:
            request (DetectionRequest): Detection request parameters
            background_tasks (BackgroundTasks): FastAPI background tasks
            
        Returns:
            DetectionResponse: Detection results with confidence scores
            
        Raises:
            HTTPException: If detection fails or invalid parameters
        """
        start_time = time.time()
        
        try:
            # Validate request
            if not request.text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Text cannot be empty"
                )
            
            # Collect detection results
            detection_results = []
            
            # Run SynthID detection if requested
            if "synthid" in [method.value for method in request.detection_methods]:
                try:
                    synthid_detector = self._get_synthid_detector()
                    if synthid_detector.is_available():
                        synthid_result = synthid_detector.detect(request.text)
                        detection_results.append(synthid_result)
                        logger.debug("SynthID detection completed")
                    else:
                        logger.warning("SynthID detector not available")
                except DetectionError as e:
                    logger.warning(f"SynthID detection failed: {e}")
                    if not e.recoverable:
                        raise HTTPException(status_code=500, detail=str(e))
                except Exception as e:
                    logger.error(f"SynthID detection error: {e}")
            
            # Run custom detection if requested
            if "custom" in [method.value for method in request.detection_methods]:
                try:
                    custom_detector = self._get_custom_detector()
                    if custom_detector.is_available():
                        custom_result = custom_detector.detect(request.text)
                        detection_results.append(custom_result)
                        logger.debug("Custom detection completed")
                    else:
                        logger.warning("Custom detector not available")
                except DetectionError as e:
                    logger.warning(f"Custom detection failed: {e}")
                    if not e.recoverable:
                        raise HTTPException(status_code=500, detail=str(e))
                except Exception as e:
                    logger.error(f"Custom detection error: {e}")
            
            # Check if we have any results
            if not detection_results:
                raise HTTPException(
                    status_code=503,
                    detail="No detection methods available"
                )
            
            # Combine results if multiple detectors were used
            if len(detection_results) > 1:
                combiner = self._get_result_combiner()
                combined_result = combiner.combine_results(detection_results, request.text)
            else:
                combined_result = detection_results[0]
            
            # Apply threshold if specified
            if request.threshold != 0.5:  # Default threshold
                combined_result.is_watermarked = combined_result.confidence_score >= request.threshold
            
            # Create response
            response = DetectionResponse(
                confidence_score=combined_result.confidence_score,
                is_watermarked=combined_result.is_watermarked,
                model_identified=combined_result.model_identified,
                detection_methods_used=[result.detection_method for result in detection_results],
                analysis_metadata=combined_result.metadata if request.include_metadata else {},
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
            
            # Update statistics
            self.total_detections += 1
            self.successful_detections += 1
            
            # Schedule background tasks
            if request.include_metadata:
                background_tasks.add_task(
                    self._save_analysis_record,
                    request.text,
                    response
                )
            
            logger.info(
                f"Detection completed: confidence={response.confidence_score:.3f}, "
                f"watermarked={response.is_watermarked}, time={response.processing_time_ms}ms"
            )
            
            return response
            
        except HTTPException:
            self.failed_detections += 1
            raise
        except Exception as e:
            self.failed_detections += 1
            logger.error(f"Detection failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Detection failed: {str(e)}"
            )
    
    async def detect_batch(
        self,
        request: BatchRequest,
        background_tasks: BackgroundTasks
    ) -> BatchResponse:
        """
        Perform batch watermark detection on multiple texts.
        
        Args:
            request (BatchRequest): Batch detection request
            background_tasks (BackgroundTasks): FastAPI background tasks
            
        Returns:
            BatchResponse: Batch detection results
        """
        start_time = time.time()
        
        try:
            results = []
            successful_count = 0
            
            for i, text in enumerate(request.texts):
                try:
                    # Create individual detection request
                    individual_request = DetectionRequest(
                        text=text,
                        detection_methods=request.detection_methods,
                        include_metadata=request.include_metadata,
                        threshold=0.5  # Use default threshold for batch
                    )
                    
                    # Perform detection
                    result = await self.detect_watermark(individual_request, background_tasks)
                    results.append(result)
                    successful_count += 1
                    
                except Exception as e:
                    logger.warning(f"Batch item {i} failed: {e}")
                    # Create error response for failed item
                    error_response = DetectionResponse(
                        confidence_score=0.0,
                        is_watermarked=False,
                        model_identified=None,
                        detection_methods_used=[],
                        analysis_metadata={"error": str(e)} if request.include_metadata else {},
                        processing_time_ms=0
                    )
                    results.append(error_response)
            
            # Calculate batch statistics
            processing_time = int((time.time() - start_time) * 1000)
            
            batch_stats = {
                "total_items": len(request.texts),
                "successful_items": successful_count,
                "failed_items": len(request.texts) - successful_count,
                "success_rate": successful_count / len(request.texts),
                "avg_confidence": sum(r.confidence_score for r in results) / len(results),
                "watermarked_count": sum(1 for r in results if r.is_watermarked)
            }
            
            response = BatchResponse(
                results=results,
                batch_stats=batch_stats,
                total_processed=len(results),
                processing_time_ms=processing_time
            )
            
            logger.info(f"Batch detection completed: {successful_count}/{len(request.texts)} successful")
            
            return response
            
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Batch detection failed: {str(e)}"
            )
    
    async def _save_analysis_record(
        self,
        input_text: str,
        result: DetectionResponse
    ) -> None:
        """
        Save analysis record for history (background task).
        
        Args:
            input_text (str): Original input text
            result (DetectionResponse): Detection result
        """
        try:
            from database.operations import AnalysisResultOperations
            
            operations = AnalysisResultOperations()
            await operations.save_analysis_result(input_text, result)
            
            logger.info(f"Analysis record saved for text length: {len(input_text)}")
        except Exception as e:
            logger.error(f"Failed to save analysis record: {e}")
    
    def get_service_stats(self) -> dict:
        """
        Get detection service statistics.
        
        Returns:
            dict: Service statistics and performance metrics
        """
        return {
            "total_detections": self.total_detections,
            "successful_detections": self.successful_detections,
            "failed_detections": self.failed_detections,
            "success_rate": (
                self.successful_detections / max(1, self.total_detections)
            ),
            "synthid_available": (
                self._synthid_detector is not None and 
                self._synthid_detector.is_available()
            ),
            "custom_available": (
                self._custom_detector is not None and 
                self._custom_detector.is_available()
            )
        }


# Global detection service instance
_detection_service: Optional[DetectionService] = None


def get_detection_service() -> DetectionService:
    """
    Get or create the global detection service instance.
    
    Returns:
        DetectionService: Global detection service
    """
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service


@router.post("/detect", response_model=DetectionResponse)
async def detect_watermark(
    request: DetectionRequest,
    background_tasks: BackgroundTasks,
    service: DetectionService = Depends(get_detection_service)
) -> DetectionResponse:
    """
    Analyze text for watermark presence.
    
    This endpoint performs watermark detection on the provided text using
    the specified detection methods. It returns confidence scores and
    classification results.
    
    Args:
        request (DetectionRequest): Detection parameters and input text
        background_tasks (BackgroundTasks): FastAPI background tasks
        service (DetectionService): Detection service dependency
        
    Returns:
        DetectionResponse: Detection results with confidence scores
    """
    return await service.detect_watermark(request, background_tasks)


@router.post("/analyze-batch", response_model=BatchResponse)
async def analyze_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    service: DetectionService = Depends(get_detection_service)
) -> BatchResponse:
    """
    Perform batch watermark detection on multiple texts.
    
    This endpoint allows processing multiple texts in a single request,
    providing efficient batch analysis capabilities.
    
    Args:
        request (BatchRequest): Batch detection parameters
        background_tasks (BackgroundTasks): FastAPI background tasks
        service (DetectionService): Detection service dependency
        
    Returns:
        BatchResponse: Batch detection results and statistics
    """
    return await service.detect_batch(request, background_tasks)


@router.get("/history", response_model=HistoryResponse)
async def get_analysis_history(
    page: int = 1,
    page_size: int = 20,
    watermarked_filter: Optional[bool] = None,
    service: DetectionService = Depends(get_detection_service)
) -> HistoryResponse:
    """
    Retrieve analysis history with pagination.
    
    This endpoint provides access to previous analysis results
    with pagination support for efficient data retrieval.
    
    Args:
        page (int): Page number (1-based)
        page_size (int): Number of records per page
        watermarked_filter (Optional[bool]): Filter by watermarked status
        service (DetectionService): Detection service dependency
        
    Returns:
        HistoryResponse: Historical analysis records
    """
    try:
        from database.operations import AnalysisResultOperations
        
        operations = AnalysisResultOperations()
        models, total_count = await operations.get_analysis_history(
            page=page,
            page_size=page_size,
            watermarked_filter=watermarked_filter
        )
        
        # Convert models to AnalysisRecord format
        records = []
        for model in models:
            record = AnalysisRecord(
                id=model.id,
                timestamp=model.timestamp,
                input_text=model.input_text,
                result=model.to_detection_response(),
                user_feedback=model.user_feedback
            )
            records.append(record)
        
        return HistoryResponse(
            records=records,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analysis history"
        )


@router.get("/stats")
async def get_detection_stats(
    service: DetectionService = Depends(get_detection_service)
) -> dict:
    """
    Get detection service statistics and performance metrics.
    
    Args:
        service (DetectionService): Detection service dependency
        
    Returns:
        dict: Service statistics and detector availability
    """
    try:
        return service.get_service_stats()
    except Exception as e:
        logger.error(f"Failed to get service stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve service statistics"
        )