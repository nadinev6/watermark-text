"""
Main application entry point for the AI Watermark Detection Tool.

This module initializes the FastAPI application, configures middleware,
and sets up the complete API structure with all endpoints and services.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from utils.config import get_config
from api.middleware import setup_middleware
from api.detection import router as detection_router
from api.test_data import router as test_data_router
from api.watermark import router as watermark_router

# main.py (update api_info endpoint)
@app.get("/api/info")
async def api_info():
    return {
        # ... existing info ...
        "endpoints": {
            # ... existing endpoints ...
            "POST /api/watermark/embed": "Embed watermark into text",
            "POST /api/watermark/extract": "Extract watermark from text"
        },
        "features": {
            "watermarking": {
                "methods": ["stegano_lsb"],
                "visibility_options": ["hidden", "visible"],
                "max_text_length": config.MAX_TEXT_LENGTH
            }
        }
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get application configuration
config = get_config()

# Initialize FastAPI application
app = FastAPI(
    title="AI Watermark Detection Tool",
    description="Detect watermarks in AI-generated text using SynthID and custom algorithms",
    version="0.1.0",
    debug=config.api.debug,
    docs_url="/docs" if config.api.debug else None,
    redoc_url="/redoc" if config.api.debug else None
)

# Configure CORS middleware (must be added before other middleware)
app.add_middleware(
    CORSMiddleware,
    **config.get_cors_settings()
)

# Set up custom middleware
setup_middleware(app)

# Include API routers
app.include_router(detection_router)
app.include_router(test_data_router)
app.include_router(watermark_router)


@app.get("/")
async def root():
    """
    Root endpoint providing basic API information.
    
    Returns:
        dict: Basic API information and status
    """
    return {
        "message": "AI Watermark Detection Tool API",
        "version": "0.1.0",
        "status": "running",
        "environment": config.environment,
        "endpoints": {
            "detection": "/api/detect",
            "batch_analysis": "/api/analyze-batch",
            "test_data_generation": "/api/generate-test-data",
            "watermark_embed": "/api/watermark/embed",
            "watermark_extract": "/api/watermark/extract",
            "watermark_validate": "/api/watermark/validate",
            "history": "/api/history",
            "stats": "/api/stats",
            "health": "/health",
            "docs": "/docs" if config.api.debug else "disabled"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and deployment.
    
    Returns:
        dict: Application health status and configuration info
    """
    try:
        # Check detector availability
        detector_status = {}
        
        try:
            from api.detection import get_detection_service
            detection_service = get_detection_service()
            service_stats = detection_service.get_service_stats()
            detector_status = {
                "synthid_available": service_stats.get("synthid_available", False),
                "custom_available": service_stats.get("custom_available", False),
                "total_detections": service_stats.get("total_detections", 0)
            }
        except Exception as e:
            logger.warning(f"Could not get detector status: {e}")
            detector_status = {"error": "Detectors not initialized"}
        
        # Check test data generator availability
        generator_status = {}
        
        try:
            from api.test_data import get_test_data_service
            test_service = get_test_data_service()
            generator_stats = test_service.get_service_stats()
            generator_status = {
                "generator_available": generator_stats.get("generator_initialized", False),
                "total_generations": generator_stats.get("total_generations", 0)
            }
        except Exception as e:
            logger.warning(f"Could not get generator status: {e}")
            generator_status = {"error": "Generator not initialized"}
        
        return {
            "status": "healthy",
            "timestamp": "unknown",
            "environment": config.environment,
            "configuration": {
                "models": {
                    "gemma": config.model.gemma_model_name,
                    "phi": config.model.phi_model_name
                },
                "detection_methods": ["synthid", "custom"],
                "api_features": {
                    "rate_limiting": True,
                    "cors_enabled": True,
                    "debug_mode": config.api.debug,
                    "api_key_required": config.api.api_key_required
                }
            },
            "services": {
                "detection": detector_status,
                "test_generation": generator_status
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "environment": config.environment
            }
        )


@app.get("/api/info")
async def api_info():
    """
    Get detailed API information and capabilities.
    
    Returns:
        dict: Comprehensive API information
    """
    return {
        "api_version": "0.1.0",
        "supported_detection_methods": [
            {
                "name": "synthid",
                "description": "Google SynthID watermark detection using Gemma models",
                "model": config.model.gemma_model_name,
                "capabilities": ["watermark_detection", "model_identification"]
            },
            {
                "name": "custom",
                "description": "Custom statistical analysis for AI-generated text detection",
                "model": config.model.phi_model_name,
                "capabilities": ["statistical_analysis", "fallback_detection"]
            }
        ],
        "supported_watermarking_methods": [
            {
                "name": "stegano_lsb",
                "description": "Hidden watermark using least significant bit steganography",
                "visibility": "completely_hidden",
                "capabilities": ["text_embedding", "text_extraction", "integrity_validation"]
            },
            {
                "name": "visible_text",
                "description": "Visible watermark as text annotation",
                "visibility": "configurable",
                "capabilities": ["text_embedding", "text_extraction", "attribution"]
            }
        ],
        "endpoints": {
            "POST /api/detect": "Analyze single text for watermarks",
            "POST /api/analyze-batch": "Batch analysis of multiple texts",
            "POST /api/generate-test-data": "Generate test datasets",
            "POST /api/watermark/embed": "Embed watermark into text",
            "POST /api/watermark/extract": "Extract watermark from text",
            "POST /api/watermark/validate": "Validate watermark integrity",
            "GET /api/watermark/methods": "Get watermarking method information",
            "GET /api/watermark/stats": "Get watermarking service statistics",
            "GET /api/history": "Retrieve analysis history",
            "GET /api/stats": "Get detection statistics",
            "GET /api/generation-stats": "Get test generation statistics",
            "GET /api/default-prompts": "Get default generation prompts"
        },
        "features": {
            "watermarking": {
                "methods": ["stegano_lsb", "visible_text"],
                "visibility_options": ["hidden", "visible"],
                "max_text_length": config.max_text_length if hasattr(config, 'max_text_length') else 50000,
                "max_watermark_length": 500
            },
            "detection": {
                "methods": ["synthid", "custom"],
                "batch_processing": True,
                "history_tracking": True
            }
        },
        "rate_limits": {
            "requests_per_minute": config.api.rate_limit_requests_per_minute,
            "max_request_size_mb": config.api.max_request_size_mb
        },
        "authentication": {
            "api_key_required": config.api.api_key_required,
            "method": "Bearer token" if config.api.api_key_required else "None"
        }
    }


# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled exceptions.
    
    Args:
        request: FastAPI request object
        exc: Exception that was raised
        
    Returns:
        JSONResponse: Standardized error response
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if config.api.debug else "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    logger.info("=" * 60)
    logger.info("AI Watermark Detection Tool Starting Up")
    logger.info("=" * 60)
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Debug mode: {config.api.debug}")
    logger.info(f"Host: {config.api.host}:{config.api.port}")
    logger.info(f"Models configured:")
    logger.info(f"  - Gemma: {config.model.gemma_model_name}")
    logger.info(f"  - Phi: {config.model.phi_model_name}")
    logger.info(f"Rate limiting: {config.api.rate_limit_requests_per_minute} req/min")
    logger.info(f"API key required: {config.api.api_key_required}")
    
    # Initialize database
    try:
        from database import initialize_database
        await initialize_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Don't fail startup, but log the error
    
    logger.info("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler."""
    logger.info("AI Watermark Detection Tool shutting down...")
    
    # Cleanup database resources
    try:
        from database import cleanup_database
        await cleanup_database()
        logger.info("Database resources cleaned up")
    except Exception as e:
        logger.warning(f"Error during database cleanup: {e}")
    
    # Cleanup detection service resources
    try:
        from api.detection import get_detection_service
        detection_service = get_detection_service()
        
        # Cleanup detector resources
        if hasattr(detection_service, '_synthid_detector') and detection_service._synthid_detector:
            detection_service._synthid_detector.cleanup()
        
        if hasattr(detection_service, '_custom_detector') and detection_service._custom_detector:
            detection_service._custom_detector.cleanup()
        
        logger.info("Detection service resources cleaned up")
        
    except Exception as e:
        logger.warning(f"Error during detection service cleanup: {e}")
    
    # Cleanup test data service resources
    try:
        from api.test_data import get_test_data_service
        test_service = get_test_data_service()
        
        # Cleanup generator resources
        if hasattr(test_service, '_generator') and test_service._generator:
            test_service._generator.cleanup()
        
        logger.info("Test data service resources cleaned up")
        
    except Exception as e:
        logger.warning(f"Error during test data service cleanup: {e}")
    
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting AI Watermark Detection Tool on {config.api.host}:{config.api.port}")
    
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level=config.log_level.lower(),
        access_log=True
    )