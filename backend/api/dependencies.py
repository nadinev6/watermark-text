"""
FastAPI dependencies for dependency injection.

This module provides dependency functions for FastAPI endpoints,
including service instances, authentication, and request validation.
"""

import logging
from typing import Optional
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from utils.config import get_config

logger = logging.getLogger(__name__)

# Security scheme for API key authentication (optional)
security = HTTPBearer(auto_error=False)


async def get_current_config():
    """
    Get current application configuration.
    
    Returns:
        AppConfig: Current application configuration
    """
    return get_config()


async def validate_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    config = Depends(get_current_config)
) -> Optional[str]:
    """
    Validate API key if authentication is required.
    
    Args:
        credentials (Optional[HTTPAuthorizationCredentials]): Bearer token credentials
        config: Application configuration
        
    Returns:
        Optional[str]: API key if valid, None if not required
        
    Raises:
        HTTPException: If API key is required but invalid
    """
    # Check if API key authentication is required
    if not config.api.api_key_required:
        return None
    
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # In a real implementation, you would validate the API key
    # against a database or configuration
    api_key = credentials.credentials
    
    # For demo purposes, accept any non-empty key
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    logger.info(f"API key validated: {api_key[:8]}...")
    return api_key


async def validate_request_size(request: Request, config = Depends(get_current_config)):
    """
    Validate request size limits.
    
    Args:
        request (Request): FastAPI request object
        config: Application configuration
        
    Raises:
        HTTPException: If request size exceeds limits
    """
    content_length = request.headers.get("content-length")
    
    if content_length:
        content_length = int(content_length)
        max_size = config.api.max_request_size_mb * 1024 * 1024  # Convert to bytes
        
        if content_length > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Request too large. Maximum size: {config.api.max_request_size_mb}MB"
            )


async def get_client_info(request: Request) -> dict:
    """
    Extract client information from request.
    
    Args:
        request (Request): FastAPI request object
        
    Returns:
        dict: Client information including IP, user agent, etc.
    """
    return {
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers)
    }


# Import detection service dependency (avoiding circular imports)
def get_detection_service():
    """
    Get detection service instance.
    
    This function is imported by the detection module to avoid circular imports.
    The actual implementation is in the detection module.
    """
    from api.detection import get_detection_service as _get_detection_service
    return _get_detection_service()


def get_test_data_service():
    """
    Get test data service instance.
    
    This function is imported by the test data module to avoid circular imports.
    The actual implementation is in the test_data module.
    """
    from api.test_data import get_test_data_service as _get_test_data_service
    return _get_test_data_service()