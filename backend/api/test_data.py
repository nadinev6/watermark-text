"""
Test data generation API endpoints.

This module provides FastAPI endpoints for generating test datasets
with watermarked and non-watermarked samples for validation purposes.
"""

import time
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from models.schemas import (
    TestDataRequest,
    TestDataResponse,
    TestSample
)
# Optional import that requires torch
try:
    from utils.test_data_generator import (
        TestDataGenerator,
        GenerationParams,
        create_test_generator
    )
    _test_data_available = True
except ImportError:
    TestDataGenerator = None
    GenerationParams = None
    create_test_generator = None
    _test_data_available = False
from utils.config import get_config

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api", tags=["test-data"])


class TestDataService:
    """
    Service class for managing test data generation operations.
    
    This class provides functionality to generate test datasets with
    both watermarked and clean samples for validation purposes.
    """
    
    def __init__(self):
        """Initialize the test data service."""
        self.config = get_config()
        
        # Test data generator (lazy loading)
        self._generator: Optional[TestDataGenerator] = None
        
        # Generation statistics
        self.total_generations = 0
        self.successful_generations = 0
        self.failed_generations = 0
        
        logger.info("Test data service initialized")
    
    def _get_generator(self) -> TestDataGenerator:
        """Get or create test data generator instance."""
        if not _test_data_available:
            raise HTTPException(
                status_code=503,
                detail="Test data generation not available - ML dependencies not installed"
            )
            
        if self._generator is None:
            try:
                self._generator = create_test_generator()
                logger.info("Test data generator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize test data generator: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="Test data generator unavailable"
                )
        return self._generator
    
    async def generate_test_data(
        self,
        request: TestDataRequest,
        background_tasks: BackgroundTasks
    ) -> TestDataResponse:
        """
        Generate test dataset with watermarked and clean samples.
        
        Args:
            request (TestDataRequest): Generation parameters
            background_tasks (BackgroundTasks): FastAPI background tasks
            
        Returns:
            TestDataResponse: Generated test samples and statistics
            
        Raises:
            HTTPException: If generation fails or invalid parameters
        """
        start_time = time.time()
        
        try:
            # Validate request
            if request.sample_count <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Sample count must be positive"
                )
            
            if not 0.0 <= request.watermarked_ratio <= 1.0:
                raise HTTPException(
                    status_code=400,
                    detail="Watermarked ratio must be between 0.0 and 1.0"
                )
            
            if not request.prompts:
                raise HTTPException(
                    status_code=400,
                    detail="Prompts list cannot be empty"
                )
            
            # Get generator
            generator = self._get_generator()
            
            # Create generation parameters
            if not _test_data_available:
                raise HTTPException(
                    status_code=503,
                    detail="Test data generation not available - ML dependencies not installed"
                )
            generation_params = GenerationParams(
                max_new_tokens=min(request.max_length, 500),  # Limit for API
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )
            
            # Generate balanced dataset
            samples = generator.create_balanced_dataset(
                total_size=request.sample_count,
                watermarked_ratio=request.watermarked_ratio,
                prompts=request.prompts,
                generation_params=generation_params
            )
            
            # Calculate statistics
            processing_time = int((time.time() - start_time) * 1000)
            watermarked_count = sum(1 for s in samples if s.is_watermarked)
            
            generation_stats = {
                "total_samples": len(samples),
                "watermarked_samples": watermarked_count,
                "clean_samples": len(samples) - watermarked_count,
                "actual_ratio": watermarked_count / len(samples) if samples else 0.0,
                "avg_text_length": sum(len(s.text) for s in samples) / len(samples) if samples else 0,
                "generation_params": generation_params.__dict__,
                "prompts_used": len(request.prompts)
            }
            
            # Create response
            response = TestDataResponse(
                samples=samples,
                generation_stats=generation_stats,
                total_samples=len(samples),
                watermarked_count=watermarked_count,
                processing_time_ms=processing_time
            )
            
            # Update statistics
            self.total_generations += 1
            self.successful_generations += 1
            
            # Schedule background cleanup
            background_tasks.add_task(self._cleanup_generator_resources)
            
            logger.info(
                f"Test data generation completed: {len(samples)} samples, "
                f"{watermarked_count} watermarked, time={processing_time}ms"
            )
            
            return response
            
        except HTTPException:
            self.failed_generations += 1
            raise
        except Exception as e:
            self.failed_generations += 1
            logger.error(f"Test data generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Test data generation failed: {str(e)}"
            )
    
    async def validate_dataset(self, samples: List[TestSample]) -> dict:
        """
        Validate generated dataset quality.
        
        Args:
            samples (List[TestSample]): Samples to validate
            
        Returns:
            dict: Validation results and statistics
        """
        try:
            generator = self._get_generator()
            validation_results = generator.validate_dataset(samples)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Dataset validation failed: {str(e)}"
            )
    
    async def _cleanup_generator_resources(self) -> None:
        """
        Clean up generator resources (background task).
        """
        try:
            if self._generator is not None:
                # Don't fully cleanup, just log
                logger.info("Generator resources cleanup scheduled")
        except Exception as e:
            logger.error(f"Failed to cleanup generator resources: {e}")
    
    def get_service_stats(self) -> dict:
        """
        Get test data service statistics.
        
        Returns:
            dict: Service statistics and performance metrics
        """
        generator_stats = {}
        if self._generator is not None:
            generator_stats = self._generator.get_generation_stats()
        
        return {
            "total_generations": self.total_generations,
            "successful_generations": self.successful_generations,
            "failed_generations": self.failed_generations,
            "success_rate": (
                self.successful_generations / max(1, self.total_generations)
            ),
            "generator_initialized": self._generator is not None,
            "generator_stats": generator_stats
        }


# Global test data service instance
_test_data_service: Optional[TestDataService] = None


def get_test_data_service() -> TestDataService:
    """
    Get or create the global test data service instance.
    
    Returns:
        TestDataService: Global test data service
    """
    global _test_data_service
    if _test_data_service is None:
        _test_data_service = TestDataService()
    return _test_data_service


@router.post("/generate-test-data", response_model=TestDataResponse)
async def generate_test_data(
    request: TestDataRequest,
    background_tasks: BackgroundTasks,
    service: TestDataService = Depends(get_test_data_service)
) -> TestDataResponse:
    """
    Generate test dataset with watermarked and clean samples.
    
    This endpoint creates a balanced dataset of text samples for testing
    watermark detection systems. It generates both watermarked samples
    using Gemma models and clean samples using Phi models.
    
    Args:
        request (TestDataRequest): Generation parameters including sample count,
                                 watermarked ratio, prompts, and max length
        background_tasks (BackgroundTasks): FastAPI background tasks
        service (TestDataService): Test data service dependency
        
    Returns:
        TestDataResponse: Generated samples with statistics and metadata
    """
    return await service.generate_test_data(request, background_tasks)


@router.post("/validate-dataset")
async def validate_dataset(
    samples: List[TestSample],
    service: TestDataService = Depends(get_test_data_service)
) -> dict:
    """
    Validate the quality of a test dataset.
    
    This endpoint analyzes a dataset of test samples and provides
    validation results including quality metrics and error detection.
    
    Args:
        samples (List[TestSample]): Test samples to validate
        service (TestDataService): Test data service dependency
        
    Returns:
        dict: Validation results with quality metrics and error reports
    """
    return await service.validate_dataset(samples)


@router.get("/generation-stats")
async def get_generation_stats(
    service: TestDataService = Depends(get_test_data_service)
) -> dict:
    """
    Get test data generation statistics and performance metrics.
    
    Args:
        service (TestDataService): Test data service dependency
        
    Returns:
        dict: Generation statistics and service performance metrics
    """
    try:
        return service.get_service_stats()
    except Exception as e:
        logger.error(f"Failed to get generation stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve generation statistics"
        )


@router.get("/default-prompts")
async def get_default_prompts() -> dict:
    """
    Get the default prompts used for test data generation.
    
    Returns:
        dict: Default prompts and evaluation prompts
    """
    try:
        if not _test_data_available:
            raise HTTPException(
                status_code=503,
                detail="Test data generation not available - ML dependencies not installed"
            )
            
        from utils.test_data_generator import create_evaluation_prompts
        
        generator = create_test_generator()
        default_prompts = generator.default_prompts
        evaluation_prompts = create_evaluation_prompts()
        
        return {
            "default_prompts": default_prompts,
            "evaluation_prompts": evaluation_prompts,
            "total_default": len(default_prompts),
            "total_evaluation": len(evaluation_prompts)
        }
        
    except Exception as e:
        logger.error(f"Failed to get default prompts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve default prompts"
        )