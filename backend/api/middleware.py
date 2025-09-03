"""
FastAPI middleware for request processing, error handling, and security.

This module provides middleware components for logging, error handling,
rate limiting, and security features.
"""

import time
import logging
import logging
from typing import Callable, Dict, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from utils.config import get_config

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging and performance monitoring.
    
    This middleware logs all incoming requests and outgoing responses,
    including timing information and error details.
    """
    
    def __init__(self, app):
        """Initialize logging middleware."""
        super().__init__(app)
        self.config = get_config()
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process request and log details.
        
        Args:
            request (Request): Incoming request
            call_next (RequestResponseEndpoint): Next middleware/endpoint
            
        Returns:
            Response: Processed response
        """
        start_time = time.time()
        
        # Extract request information
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        url = str(request.url)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request
        logger.info(
            f"Request: {method} {url} from {client_ip} "
            f"User-Agent: {user_agent[:100]}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code} for {method} {url} "
                f"in {process_time:.3f}s"
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Calculate processing time for errors
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Error: {str(e)} for {method} {url} "
                f"in {process_time:.3f}s"
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(e) if self.config.api.debug else "An error occurred",
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers={"X-Process-Time": str(process_time)}
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests per client.
    
    This middleware implements a sliding window rate limiter to prevent
    abuse and ensure fair usage of the API.
    """
    
    def __init__(self, app):
        """Initialize rate limiting middleware."""
        super().__init__(app)
        self.config = get_config()
        
        # Rate limiting storage (in production, use Redis or similar)
        self.client_requests: Dict[str, deque] = defaultdict(deque)
        self.rate_limit = self.config.api.rate_limit_requests_per_minute
        self.window_size = timedelta(minutes=1)
        
        logger.info(f"Rate limiting enabled: {self.rate_limit} requests per minute")
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Check rate limits and process request.
        
        Args:
            request (Request): Incoming request
            call_next (RequestResponseEndpoint): Next middleware/endpoint
            
        Returns:
            Response: Processed response or rate limit error
        """
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health checks
        if request.url.path in ["/", "/health"]:
            return await call_next(request)
        
        # Check rate limit
        now = datetime.utcnow()
        client_requests = self.client_requests[client_ip]
        
        # Remove old requests outside the window
        while client_requests and client_requests[0] < now - self.window_size:
            client_requests.popleft()
        
        # Check if rate limit exceeded
        if len(client_requests) >= self.rate_limit:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.rate_limit} requests per minute",
                    "retry_after": 60,
                    "timestamp": now.isoformat()
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int((now + self.window_size).timestamp()))
                }
            )
        
        # Add current request to tracking
        client_requests.append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.rate_limit - len(client_requests))
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int((now + self.window_size).timestamp()))
        
        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for security headers and basic security measures.
    
    This middleware adds security headers and performs basic security
    checks on incoming requests.
    """
    
    def __init__(self, app):
        """Initialize security middleware."""
        super().__init__(app)
        self.config = get_config()
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Add security headers and process request.
        
        Args:
            request (Request): Incoming request
            call_next (RequestResponseEndpoint): Next middleware/endpoint
            
        Returns:
            Response: Response with security headers
        """
        # Basic security checks
        content_type = request.headers.get("content-type", "")
        
        # Check for suspicious content types
        if request.method in ["POST", "PUT", "PATCH"]:
            if content_type and not any(
                allowed in content_type.lower()
                for allowed in ["application/json", "multipart/form-data", "application/x-www-form-urlencoded"]
            ):
                logger.warning(f"Suspicious content type: {content_type}")
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
        }
        
        # Add headers to response
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for global error handling and standardized error responses.
    
    This middleware catches unhandled exceptions and returns standardized
    error responses with appropriate status codes and messages.
    """
    
    def __init__(self, app):
        """Initialize error handling middleware."""
        super().__init__(app)
        self.config = get_config()
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Handle errors and return standardized responses.
        
        Args:
            request (Request): Incoming request
            call_next (RequestResponseEndpoint): Next middleware/endpoint
            
        Returns:
            Response: Processed response or error response
        """
        try:
            return await call_next(request)
            
        except HTTPException:
            # Re-raise HTTP exceptions (handled by FastAPI)
            raise
            
        except ValueError as e:
            # Handle validation errors
            logger.warning(f"Validation error: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Validation error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except PermissionError as e:
            # Handle permission errors
            logger.warning(f"Permission error: {e}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Permission denied",
                    "message": "Insufficient permissions",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except FileNotFoundError as e:
            # Handle file not found errors
            logger.warning(f"File not found: {e}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Resource not found",
                    "message": "The requested resource was not found",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except TimeoutError as e:
            # Handle timeout errors
            logger.error(f"Timeout error: {e}")
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Request timeout",
                    "message": "The request took too long to process",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            # Handle all other exceptions
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            
            error_message = str(e) if self.config.api.debug else "An internal error occurred"
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": error_message,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )


def setup_middleware(app):
    """
    Set up all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Add middleware in reverse order (last added is executed first)
    
    # Error handling (outermost)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Security headers
    app.add_middleware(SecurityMiddleware)
    
    # Rate limiting
    app.add_middleware(RateLimitMiddleware)
    
    # Logging (innermost, closest to endpoints)
    app.add_middleware(LoggingMiddleware)
    
    logger.info("All middleware configured successfully")