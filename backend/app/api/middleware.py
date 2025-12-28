"""
Custom FastAPI middleware for request processing, logging, and monitoring.
"""
import time
import json
import uuid
from typing import Callable, Optional, Dict, Any
from datetime import datetime

from fastapi import Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
import structlog

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.security import verify_token

logger = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add unique request ID to each request.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured logging of requests and responses.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timer
        start_time = time.time()
        
        # Get request details
        request_id = getattr(request.state, "request_id", "unknown")
        client_ip = request.client.host if request.client else "0.0.0.0"
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Extract user ID from token if present
        user_id = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                payload = verify_token(token)
                if payload:
                    user_id = payload.get("sub")
            except Exception:
                pass
        
        # Log request
        logger.info(
            "request.start",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=client_ip,
            user_id=user_id,
            user_agent=user_agent,
            content_type=request.headers.get("Content-Type"),
            content_length=request.headers.get("Content-Length"),
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "request.complete",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time=round(process_time * 1000, 2),  # in milliseconds
                response_size=response.headers.get("Content-Length"),
            )
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                "request.error",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                error_type=type(e).__name__,
                error_message=str(e),
                process_time=round(process_time * 1000, 2),
            )
            raise


class DatabaseSessionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to attach database session to request state.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Create database session
        db = SessionLocal()
        request.state.db = db
        
        try:
            # Process request
            response = await call_next(request)
            
            # Commit transaction if no error
            if response.status_code < 400:
                db.commit()
            else:
                db.rollback()
            
            return response
            
        except Exception as e:
            # Rollback on error
            db.rollback()
            raise
            
        finally:
            # Close database session
            db.close()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }
        
        # Add CSP if configured
        if settings.CSP_DIRECTIVES:
            response.headers["Content-Security-Policy"] = settings.CSP_DIRECTIVES
        
        # Add all security headers
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add rate limit headers to responses.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add rate limit headers if they exist in request state
        if hasattr(request.state, "rate_limit_headers"):
            for header, value in request.state.rate_limit_headers.items():
                response.headers[header] = value
        
        return response


class MaintenanceModeMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enable maintenance mode.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if settings.MAINTENANCE_MODE:
            # Allow health checks and admin endpoints
            if request.url.path in ["/health", "/api/v1/health", "/admin/health"]:
                return await call_next(request)
            
            # Return maintenance response
            return Response(
                content=json.dumps({
                    "error": "Maintenance mode",
                    "message": "Service is temporarily unavailable for maintenance",
                    "estimated_recovery": settings.MAINTENANCE_ESTIMATED_RECOVERY,
                }),
                status_code=503,
                media_type="application/json",
                headers={
                    "Retry-After": "300",  # 5 minutes
                    "X-Maintenance-Mode": "true",
                }
            )
        
        return await call_next(request)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit request body size.
    """
    def __init__(self, app, max_size_mb: int = 10):
        super().__init__(app)
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("Content-Length")
        
        if content_length and int(content_length) > self.max_size_bytes:
            return Response(
                content=json.dumps({
                    "error": "Request too large",
                    "message": f"Request body exceeds {self.max_size_bytes // (1024*1024)}MB limit",
                    "max_size_mb": self.max_size_bytes // (1024*1024),
                }),
                status_code=413,
                media_type="application/json"
            )
        
        return await call_next(request)


class CacheControlMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add cache control headers.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Skip cache for certain endpoints
        if request.url.path.startswith("/api/") or request.url.path.startswith("/auth/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        else:
            # Default cache for static assets
            response.headers["Cache-Control"] = "public, max-age=3600"
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle and format errors consistently.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
            
        except HTTPException as e:
            # FastAPI HTTP exceptions
            return Response(
                content=json.dumps({
                    "error": {
                        "code": e.status_code,
                        "message": e.detail,
                        "type": type(e).__name__,
                    }
                }),
                status_code=e.status_code,
                media_type="application/json",
                headers=getattr(e, "headers", None)
            )
            
        except Exception as e:
            # Unexpected errors
            logger.error(
                "unexpected_error",
                request_id=getattr(request.state, "request_id", "unknown"),
                url=str(request.url),
                method=request.method,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            
            # Return generic error in production, detailed in development
            if settings.DEBUG:
                error_detail = str(e)
            else:
                error_detail = "An unexpected error occurred"
            
            return Response(
                content=json.dumps({
                    "error": {
                        "code": 500,
                        "message": "Internal server error",
                        "detail": error_detail,
                        "request_id": getattr(request.state, "request_id", "unknown"),
                    }
                }),
                status_code=500,
                media_type="application/json"
            )


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect request metrics.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timer
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        process_time = time.time() - start_time
        
        # Store metrics (could be sent to Prometheus, statsd, etc.)
        metrics = {
            "endpoint": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "process_time": process_time,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Log metrics
        logger.info("request.metrics", **metrics)
        
        # Add metrics to response headers
        response.headers["X-Response-Time"] = str(round(process_time * 1000, 2))
        
        return response


class CORSMiddlewareCustom(CORSMiddleware):
    """
    Custom CORS middleware with additional configuration.
    """
    def __init__(self, app):
        super().__init__(
            app,
            allow_origins=settings.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "X-Request-ID",
                "X-Requested-With",
                "X-CSRF-Token",
                "X-API-Key",
            ],
            expose_headers=[
                "X-Request-ID",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
                "X-Process-Time",
                "X-Response-Time",
            ],
            max_age=600,  # 10 minutes
        )


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API keys for specific endpoints.
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if endpoint requires API key
        if request.url.path.startswith("/api/v1/webhooks/"):
            api_key = request.headers.get("X-API-Key")
            
            if not api_key:
                return Response(
                    content=json.dumps({
                        "error": "API key required",
                        "message": "Missing X-API-Key header"
                    }),
                    status_code=401,
                    media_type="application/json"
                )
            
            # Validate API key (simplified - in production, check against database)
            if not self.is_valid_api_key(api_key):
                return Response(
                    content=json.dumps({
                        "error": "Invalid API key",
                        "message": "The provided API key is invalid or expired"
                    }),
                    status_code=401,
                    media_type="application/json"
                )
        
        return await call_next(request)
    
    def is_valid_api_key(self, api_key: str) -> bool:
        """
        Validate API key.
        In production, check against database or external service.
        """
        # Placeholder - implement actual validation
        return True


def setup_middleware(app):
    """
    Configure all middleware for the FastAPI application.
    """
    # Order matters! Middleware are executed in reverse order
    
    # 1. Error handling (first to catch all errors)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # 2. Maintenance mode
    app.add_middleware(MaintenanceModeMiddleware)
    
    # 3. Request size limit
    app.add_middleware(RequestSizeLimitMiddleware, max_size_mb=10)
    
    # 4. API Key validation
    app.add_middleware(APIKeyMiddleware)
    
    # 5. CORS
    app.add_middleware(CORSMiddlewareCustom)
    
    # 6. Trusted hosts
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)
    
    # 7. GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 8. Sessions (if needed)
    if settings.ENABLE_SESSIONS:
        app.add_middleware(
            SessionMiddleware,
            secret_key=settings.SECRET_KEY,
            session_cookie="session",
            max_age=14 * 24 * 60 * 60,  # 14 days
        )
    
    # 9. Request ID
    app.add_middleware(RequestIDMiddleware)
    
    # 10. Logging
    app.add_middleware(LoggingMiddleware)
    
    # 11. Database session
    app.add_middleware(DatabaseSessionMiddleware)
    
    # 12. Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # 13. Rate limit headers
    app.add_middleware(RateLimitHeadersMiddleware)
    
    # 14. Cache control
    app.add_middleware(CacheControlMiddleware)
    
    # 15. Metrics (last to capture everything)
    app.add_middleware(MetricsMiddleware)
    
    return app


# Export middleware setup function
__all__ = ["setup_middleware"]